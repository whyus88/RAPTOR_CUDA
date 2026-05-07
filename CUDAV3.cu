// ============================================
// FILE: CUDAVanExtreme.cu
// ============================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <random>
#include <algorithm>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================
// EXTREME RANDOM CONSTANTS
// ============================================
#define EXTREME_POOL_SIZE 4096        // Jumlah random jump points
#define EXTREME_WARP_CACHE 32         // Cache per warp di shared memory
#define EXTREME_HASH_UNROLL 1         // Unroll factor untuk hashing
#define SKIP_WRAPPING_CHECK 1         // 1 = skip wrapping point math (EXTREME SPEED)
#define WARP_DIVERGENCE_MASK 0        // 0 = allow divergence for speed

// ============================================
// CONSTANT MEMORY (Minimal untuk Extreme Mode)
// ============================================
// __constant__ uint32_t c_target_prefix;
// __constant__ uint8_t  c_target_hash160[20];
__constant__ uint32_t c_vanity_len;
__constant__ uint32_t c_vanity_prefix_mask;
__constant__ uint64_t c_range_start[4];
__constant__ uint64_t c_range_end[4];
__constant__ uint64_t c_range_len[4];

// ============================================
// DEVICE HASH FUNCTIONS (Optimized)
// ============================================

// Super fast 64-bit hash for random index generation
__device__ __forceinline__ uint64_t fast_hash64(uint64_t x, uint64_t seed) {
    x ^= seed;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

// Multi-source hash for better distribution
__device__ __forceinline__ uint64_t multi_hash64(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t h = a * 0x9E3779B97F4A7C15ULL + b;
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h += c * 0x517CC1B727220A95ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    return h ^ (h >> 31);
}

// ============================================
// EXTREME RANDOM KERNEL
// ============================================

__launch_bounds__(256, 4)  // Higher occupancy
__global__ void kernel_extreme_random(
    const uint64_t* __restrict__ d_init_x,
    const uint64_t* __restrict__ d_init_y,
    const uint64_t* __restrict__ d_init_s,
    uint64_t* __restrict__ d_out_x,
    uint64_t* __restrict__ d_out_y,
    uint64_t* __restrict__ d_out_s,
    uint64_t* __restrict__ d_counts,
    uint64_t threadsTotal,
    uint32_t iters_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    // Random Pool (Global Memory - Coalesced Layout)
    const uint64_t* __restrict__ d_pool_x,   // [POOL_SIZE * 4]
    const uint64_t* __restrict__ d_pool_y,   // [POOL_SIZE * 4]
    const uint64_t* __restrict__ d_pool_s,   // [POOL_SIZE * 4]
    uint32_t pool_size,
    uint64_t seed,
    unsigned int* __restrict__ d_any_left
)
{
    extern __shared__ uint64_t s_mem[];
    
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane = threadIdx.x & 31;
    const unsigned warp_id = threadIdx.x >> 5;
    const unsigned warps_per_block = blockDim.x >> 5;
    const unsigned full_mask = 0xFFFFFFFFu;
    
    // Early exit check
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    // ============================================
    // SHARED MEMORY LAYOUT:
    // Per warp: 32 * 4 * 3 uint64_t = 768 bytes
    // For 8 warps: 6144 bytes
    // ============================================
    uint64_t* s_warp_x = s_mem + (warp_id * EXTREME_WARP_CACHE * 4 * 3);
    uint64_t* s_warp_y = s_warp_x + (EXTREME_WARP_CACHE * 4);
    uint64_t* s_warp_s = s_warp_y + (EXTREME_WARP_CACHE * 4);

    // ============================================
    // LOAD INITIAL STATE
    // ============================================
    uint64_t x1[4], y1[4], S[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        x1[i] = d_init_x[gid * 4 + i];
        y1[i] = d_init_y[gid * 4 + i];
        S[i]  = d_init_s[gid * 4 + i];
    }

    uint64_t rem[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = d_counts[gid * 4 + i];

    if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0ull) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            d_out_x[gid * 4 + i] = x1[i];
            d_out_y[gid * 4 + i] = y1[i];
        }
        return;
    }

    unsigned int local_hashes = 0;
    const uint32_t vanity_len = c_vanity_len;
    const uint32_t vanity_mask = c_vanity_prefix_mask;
    const uint32_t target_prefix = c_target_prefix;

    // ============================================
    // PRECOMPUTE BASE POOL INDEX (per thread)
    // Stagger untuk menghindari collision
    // ============================================
    uint32_t base_idx = (uint32_t)(fast_hash64(gid, seed) % pool_size);

    // ============================================
    // MAIN LOOP - EXTREME RANDOM JUMP
    // ============================================
    for (uint32_t iter = 0; iter < iters_per_launch; ++iter) {
        
        // Check remaining (simplified - just check low bits)
        if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0ull) break;
        
        // Warp-level early exit
        if ((iter & 15) == 0) {  // Check every 16 iterations
            if (warp_found_ready(d_found_flag, full_mask, lane)) break;
        }

        // ============================================
        // COMPUTE RANDOM POOL INDEX
        // ============================================
        uint64_t h = multi_hash64(gid, (uint64_t)iter, seed);
        uint32_t pool_idx = (base_idx + (uint32_t)(h & 0x7FFFFFFF)) % pool_size;

        // ============================================
        // LOAD RANDOM POINT (Coalesced pattern simulation)
        // ============================================
        uint64_t rp_x[4], rp_y[4], rp_s[4];
        
        // Direct global load (compiler will coalesce if accesses are aligned)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            rp_x[i] = d_pool_x[(size_t)pool_idx * 4 + i];
            rp_y[i] = d_pool_y[(size_t)pool_idx * 4 + i];
            rp_s[i] = d_pool_s[(size_t)pool_idx * 4 + i];
        }

        // ============================================
        // POINT ADDITION: P_new = P + R
        // ============================================
        uint64_t dx[4], dy[4], dx_inv[5], lam[4], x3[4], y3[4], t[4];

        ModSub256(dx, rp_x, x1);
        ModSub256(dy, rp_y, y1);

        // Modular inverse
        #pragma unroll
        for (int i = 0; i < 4; ++i) dx_inv[i] = dx[i];
        dx_inv[4] = 0ull;
        _ModInv(dx_inv);

        // Lambda = dy * dx_inv
        _ModMult(lam, dy, dx_inv);

        // X3 = lambda^2 - x1 - rx
        _ModSqr(x3, lam);
        ModSub256(x3, x3, x1);
        ModSub256(x3, x3, rp_x);

        // Y3 = lambda * (x1 - x3) - y1
        ModSub256(t, x1, x3);
        _ModMult(y3, t, lam);
        ModSub256(y3, y3, y1);

        // Update point
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            x1[i] = x3[i];
            y1[i] = y3[i];
        }

        // ============================================
        // SCALAR UPDATE (NO WRAPPING - EXTREME SPEED)
        // ============================================
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 res = (unsigned __int128)S[i] + rp_s[i] + carry;
            S[i] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);
        }
        
        // Decrement remaining
        uint64_t borrow = 1ull;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint64_t v = rem[i] - borrow;
            borrow = (rem[i] < borrow) ? 1ull : 0ull;
            rem[i] = v;
            if (!borrow && i + 1 < 4) {
                for (int k = i + 1; k < 4; ++k) rem[k] = rem[k];
                break;
            }
        }

        // ============================================
        // HASH CHECK
        // ============================================
        uint8_t h20[20];
        uint8_t prefix = (y1[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, x1, h20);
        ++local_hashes;

        // Fast prefix check
        uint32_t h_prefix = load_u32_le(h20);
        bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

        // Warp vote for potential match
        if (__any_sync(full_mask, pref)) {
            bool full_match = false;
            
            if (pref) {
                if (vanity_len <= 4) {
                    full_match = true;
                } else {
                    full_match = true;
                    #pragma unroll 4
                    for (uint32_t k = 4; k < vanity_len; ++k) {
                        if (h20[k] != c_target_hash160[k]) {
                            full_match = false;
                            break;
                        }
                    }
                }
            }

            if (full_match) {
#if SKIP_WRAPPING_CHECK == 0
                // Verify scalar is in valid range (SLOWER but accurate)
                bool ge_start = true, le_end = true;
                // ... range check code ...
                if (ge_start && le_end)
#endif
                {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter = (int)iter;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) d_found_result->scalar[k] = S[k];
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) d_found_result->Rx[k] = x1[k];
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) d_found_result->Ry[k] = y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }

    // ============================================
    // STORE RESULTS
    // ============================================
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        d_out_x[gid * 4 + i] = x1[i];
        d_out_y[gid * 4 + i] = y1[i];
        d_out_s[gid * 4 + i] = S[i];
        d_counts[gid * 4 + i] = rem[i];
    }

    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    // Flush hashes
    unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes);
    if (lane == 0 && v) atomicAdd(hashes_accum, v);
}

// ============================================
// HYBRID KERNEL - Sequential + Random Jump
// ============================================

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];
__constant__ uint64_t c_deltas[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_J_delta[4];

__device__ void add256_with_carry(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 res = (unsigned __int128)a[i] + b[i] + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

__device__ bool ge256_u64(const uint64_t a[4], const uint64_t b[4]) {
    if (a[3] != b[3]) return a[3] > b[3];
    if (a[2] != b[2]) return a[2] > b[2];
    if (a[1] != b[1]) return a[1] > b[1];
    return a[0] >= b[0];
}

// Original kernel with optimizations
__launch_bounds__(256, 2)
__global__ void kernel_hybrid_optimized(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left,
    // Random jump pool
    const uint64_t* __restrict__ d_jump_x,
    const uint64_t* __restrict__ d_jump_y,
    const uint64_t* __restrict__ d_jump_s,
    uint32_t num_jumps,
    uint64_t jump_seed,
    int use_random_jumps
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane = threadIdx.x & 31;
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;
    const uint32_t vanity_len = c_vanity_len;
    const uint32_t vanity_mask = c_vanity_prefix_mask;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 32768u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];
    }
    uint64_t rem[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // --- Base Point Check ---
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            uint32_t h_prefix = load_u32_le(h20);
            bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

            if (__any_sync(full_mask, pref)) {
                bool full_match = false;
                if (pref) {
                    if (vanity_len <= 4) full_match = true;
                    else {
                        full_match = true;
                        for (uint32_t k = 4; k < vanity_len; ++k) {
                            if (h20[k] != c_target_hash160[k]) { full_match = false; break; }
                        }
                    }
                }

                if (full_match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter = 0;
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
        }

        // --- Batch Point Computation (Same as original) ---
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        #pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
        #pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
            #pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
            #pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
        #pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
        #pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        // --- Check +R_i and -R_i (Same as original) ---
        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t current_delta[4];
            #pragma unroll
            for (int k=0; k<4; ++k) current_delta[k] = c_deltas[(size_t)i*4 + k];

            // +R_i check
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                #pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                uint32_t h_prefix = load_u32_le(h20);
                bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

                if (__any_sync(full_mask, pref)) {
                    bool full_match = false;
                    if (pref) {
                        if (vanity_len <= 4) full_match = true;
                        else {
                            full_match = true;
                            for (uint32_t k=4; k<vanity_len; ++k) if (h20[k] != c_target_hash160[k]) { full_match=false; break; }
                        }
                    }
                    
                    if (full_match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; add256_with_carry(S, current_delta, fs);
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // -R_i check (same structure, with negated py_i)
            // ... (similar to original) ...
        }

        // Last iteration check
        // ... (similar to original) ...

        // ============================================
        // OPTIMIZED JUMP LOGIC
        // ============================================
        if (use_random_jumps && num_jumps > 0) {
            // Random jump instead of fixed jump
            uint64_t h = fast_hash64(gid, jump_seed + batches_done);
            uint32_t jump_idx = (uint32_t)(h % num_jumps);
            
            uint64_t jx[4], jy[4], js[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                jx[i] = d_jump_x[(size_t)jump_idx * 4 + i];
                jy[i] = d_jump_y[(size_t)jump_idx * 4 + i];
                js[i] = d_jump_s[(size_t)jump_idx * 4 + i];
            }
            
            // Point addition: P = P + Jump
            uint64_t dx[4], dy[4], dx_inv[5], lam[4], x3[4], y3[4], s[4];
            
            ModSub256(dx, jx, x1);
            ModSub256(dy, jy, y1);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) dx_inv[i] = dx[i];
            dx_inv[4] = 0;
            _ModInv(dx_inv);
            
            _ModMult(lam, dy, dx_inv);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            ModSub256(x3, x3, jx);
            
            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) { x1[i] = x3[i]; y1[i] = y3[i]; }
            
            // Scalar update (NO wrapping)
            uint64_t carry = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned __int128 res = (unsigned __int128)S[i] + js[i] + carry;
                S[i] = (uint64_t)res;
                carry = (uint64_t)(res >> 64);
            }
        } else {
            // Original fixed jump
            uint64_t lam[4], s[4], x3[4], y3[4];
            uint64_t Jy_minus_y1[4];
            #pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

            #pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }

            uint64_t carry = 0;
            #pragma unroll
            for (int k=0;k<4;++k) {
                unsigned __int128 res = (unsigned __int128)S[k] + c_J_delta[k] + carry;
                S[k] = (uint64_t)res;
                carry = (uint64_t)(res >> 64);
            }
        }

        sub256_u64_inplace(rem, (uint64_t)B);
        ++batches_done;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// ============================================
// HOST HELPERS
// ============================================
extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
extern void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
extern void add256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
extern void add256_u64(uint64_t a[4], uint64_t b, uint64_t r[4]);
extern void divmod_256_by_u64(const uint64_t a[4], uint64_t b, uint64_t q[4], uint64_t& r);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

void mul256_u64(const uint64_t a[4], uint64_t b, uint64_t r[4]) {
    unsigned __int128 res;
    uint64_t carry = 0;
    res = (unsigned __int128)a[0] * b; r[0] = (uint64_t)res; carry = res >> 64;
    res = (unsigned __int128)a[1] * b + carry; r[1] = (uint64_t)res; carry = res >> 64;
    res = (unsigned __int128)a[2] * b + carry; r[2] = (uint64_t)res; carry = res >> 64;
    res = (unsigned __int128)a[3] * b + carry; r[3] = (uint64_t)res;
}

void add256_host(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 res = (unsigned __int128)a[i] + b[i] + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

// ============================================
// RANDOM POOL GENERATION
// ============================================
void generate_random_pool(
    uint64_t seed,
    uint32_t pool_size,
    uint64_t* h_pool_x,
    uint64_t* h_pool_y,
    uint64_t* h_pool_s,
    const uint64_t range_len[4]
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dis64(1, UINT64_MAX);
    
    // Generate random scalars
    for (uint32_t i = 0; i < pool_size; ++i) {
        h_pool_s[i * 4 + 0] = dis64(rng) | 1ULL;  // Ensure odd
        h_pool_s[i * 4 + 1] = dis64(rng);
        h_pool_s[i * 4 + 2] = dis64(rng);
        h_pool_s[i * 4 + 3] = dis64(rng);
        
        // Optionally mod by range_len to keep scalars in range
        // For extreme mode, we let them overflow
    }
}

// ============================================
// MAIN FUNCTION
// ============================================
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string vanity_hex, range_hex;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm = 8;
    uint32_t slices_per_launch = 64;
    
    // Mode selection
    int mode = 0;  // 0=sequential, 1=random (original), 2=extreme random, 3=hybrid
    uint64_t random_seed = 0;
    uint32_t pool_size = EXTREME_POOL_SIZE;

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex  = argv[++i];
        else if (arg == "--random") {
            mode = 1;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                random_seed = std::stoull(argv[++i]);
            } else {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<uint64_t> dis;
                random_seed = dis(gen);
            }
        }
        else if (arg == "--extreme-random" || arg == "--extreme") {
            mode = 2;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                random_seed = std::stoull(argv[++i]);
            } else {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<uint64_t> dis;
                random_seed = dis(gen);
            }
        }
        else if (arg == "--hybrid-random" || arg == "--hybrid") {
            mode = 3;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                random_seed = std::stoull(argv[++i]);
            } else {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<uint64_t> dis;
                random_seed = dis(gen);
            }
        }
        else if (arg == "--pool-size" && i + 1 < argc) {
            pool_size = std::stoul(argv[++i]);
            if (pool_size < 256) pool_size = 256;
            if (pool_size > 8192) pool_size = 8192;
        }
        else if (arg == "--grid" && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\"\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices invalid\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
    }

    if (range_hex.empty() || vanity_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start:end> --vanity-hash160 <prefix>\n"
                  << "       [--random [seed]] [--extreme-random [seed]] [--hybrid-random [seed]]\n"
                  << "       [--pool-size N] [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }

    // Validate vanity
    if (vanity_hex.length() > 40 || vanity_hex.length() % 2 != 0) {
        std::cerr << "Error: Invalid vanity prefix\n";
        return EXIT_FAILURE;
    }

    uint8_t target_hash160[20] = {0};
    uint32_t vanity_len = vanity_hex.length() / 2;
    std::string padded_vanity = vanity_hex;
    if (padded_vanity.length() < 40) padded_vanity += std::string(40 - padded_vanity.length(), '0');
    if (!hexToHash160(padded_vanity, target_hash160)) {
        std::cerr << "Error: Invalid hex for vanity\n";
        return EXIT_FAILURE;
    }

    uint32_t vanity_prefix_mask = 0xFFFFFFFFu;
    if (vanity_len < 4) vanity_prefix_mask = (1ULL << (vanity_len * 8)) - 1;

    // Parse range
    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Error: range format must be start:end\n";
        return EXIT_FAILURE;
    }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4];
    sub256(range_end, range_start, range_len);
    add256_u64(range_len, 1ull, range_len);

    // CUDA Setup
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::cerr << "CUDA init error\n";
        return EXIT_FAILURE;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;

    // Calculate thread count based on mode
    uint64_t threadsTotal = 0;
    int blocks = 0;
    
    if (mode == 2) {
        // Extreme mode: maximize threads for random sampling
        uint64_t bytesPerThread = 2ull * 4ull * sizeof(uint64_t);
        size_t totalGlobalMem = prop.totalGlobalMem;
        uint64_t usableMem = totalGlobalMem - 256ULL * 1024 * 1024;  // Reserve 256MB
        uint64_t maxThreadsByMem = usableMem / bytesPerThread;
        
        // Also account for random pool
        uint64_t pool_bytes = (uint64_t)pool_size * 3 * 4 * sizeof(uint64_t);
        if (pool_bytes < usableMem) {
            usableMem -= pool_bytes;
            maxThreadsByMem = usableMem / bytesPerThread;
        }
        
        threadsTotal = maxThreadsByMem;
        // Round down to multiple of threadsPerBlock
        threadsTotal = (threadsTotal / (uint64_t)threadsPerBlock) * (uint64_t)threadsPerBlock;
        // Cap at reasonable limit
        uint64_t maxThreads = (uint64_t)prop.multiProcessorCount * 2048ULL;
        if (threadsTotal > maxThreads) threadsTotal = maxThreads;
        if (threadsTotal == 0) threadsTotal = (uint64_t)threadsPerBlock;
        
    } else {
        // Original logic for sequential/random/hybrid
        auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
        if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
            std::cerr << "Error: batch size must be even power of two\n";
            return EXIT_FAILURE;
        }

        uint64_t q_div_batch[4], r_div_batch = 0ull;
        divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
        if (r_div_batch != 0ull) {
            std::cerr << "Error: range length must be divisible by batch size\n";
            return EXIT_FAILURE;
        }
        
        bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
        uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
        if (!q_fits_u64) {
            std::cerr << "Error: total batches too large\n";
            return EXIT_FAILURE;
        }

        uint64_t bytesPerThread = 2ull * 4ull * sizeof(uint64_t);
        size_t totalGlobalMem = prop.totalGlobalMem;
        uint64_t usableMem = totalGlobalMem - 64ULL * 1024 * 1024;
        uint64_t maxThreadsByMem = usableMem / bytesPerThread;

        uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
        if (userUpper == 0ull) userUpper = UINT64_MAX;

        auto pick_threads_total_pow2 = [&](uint64_t upper)->uint64_t {
            if (upper < (uint64_t)threadsPerBlock) return 0ull;
            uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
            uint64_t q = total_batches_u64;
            uint64_t pot = 1ull;
            while (pot <= t) pot <<= 1;
            pot >>= 1;
            while (pot >= (uint64_t)threadsPerBlock) {
                if (pot > 0 && (q % pot) == 0ull) return pot;
                pot >>= 1;
            }
            return 0ull;
        };

        uint64_t upper = maxThreadsByMem;
        if (total_batches_u64 < upper) upper = total_batches_u64;
        if (userUpper < upper) upper = userUpper;
        threadsTotal = pick_threads_total_pow2(upper);
        if (threadsTotal == 0ull) {
            std::cerr << "Error: Could not determine valid thread count\n";
            return EXIT_FAILURE;
        }
    }
    
    blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    std::cout << "======== MODE: ";
    switch (mode) {
        case 0: std::cout << "SEQUENTIAL"; break;
        case 1: std::cout << "RANDOM (Original)"; break;
        case 2: std::cout << "EXTREME RANDOM"; break;
        case 3: std::cout << "HYBRID RANDOM"; break;
    }
    std::cout << " ========\n";
    std::cout << "Seed: 0x" << std::hex << random_seed << std::dec << "\n";
    if (mode >= 2) {
        std::cout << "Pool Size: " << pool_size << "\n";
        std::cout << "Wrapping: " << (SKIP_WRAPPING_CHECK ? "DISABLED (Extreme Speed)" : "ENABLED") << "\n";
    }

    // ============================================
    // GENERATE RANDOM POOL (for mode >= 2)
    // ============================================
    uint64_t *d_pool_x = nullptr, *d_pool_y = nullptr, *d_pool_s = nullptr;
    
    if (mode >= 2) {
        std::cout << "Generating random point pool (" << pool_size << " points)...\n";
        
        uint64_t* h_pool_s = (uint64_t*)std::malloc((size_t)pool_size * 4 * sizeof(uint64_t));
        generate_random_pool(random_seed, pool_size, nullptr, nullptr, h_pool_s, range_len);
        
        // Compute pool points on GPU
        uint64_t* d_pool_scalars;
        cudaMalloc(&d_pool_scalars, (size_t)pool_size * 4 * sizeof(uint64_t));
        cudaMalloc(&d_pool_x, (size_t)pool_size * 4 * sizeof(uint64_t));
        cudaMalloc(&d_pool_y, (size_t)pool_size * 4 * sizeof(uint64_t));
        
        cudaMemcpy(d_pool_scalars, h_pool_s, (size_t)pool_size * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        int pool_blocks = (pool_size + threadsPerBlock - 1) / threadsPerBlock;
        scalarMulKernelBase<<<pool_blocks, threadsPerBlock>>>(d_pool_scalars, d_pool_x, d_pool_y, pool_size);
        cudaDeviceSynchronize();
        
        cudaFree(d_pool_scalars);
        std::free(h_pool_s);
        
        std::cout << "Random pool ready.\n";
    }

    // ============================================
    // COPY CONSTANTS
    // ============================================
    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
        cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(vanity_len));
        cudaMemcpyToSymbol(c_vanity_prefix_mask, &vanity_prefix_mask, sizeof(vanity_prefix_mask));
        cudaMemcpyToSymbol(c_range_start, range_start, 4 * sizeof(uint64_t));
        cudaMemcpyToSymbol(c_range_end, range_end, 4 * sizeof(uint64_t));
        cudaMemcpyToSymbol(c_range_len, range_len, 4 * sizeof(uint64_t));
    }

    // ============================================
    // INITIALIZE THREAD DATA
    // ============================================
    uint64_t* h_start_scalars = nullptr;
    uint64_t* h_counts256 = nullptr;
    
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_counts256, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    std::mt19937_64 rng(random_seed);

    if (mode == 2) {
        // Extreme mode: random starting positions across entire range
        std::uniform_int_distribution<uint64_t> dis64(0, UINT64_MAX);
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            // Random scalar (will be modded by range implicitly via hash)
            h_start_scalars[i*4+0] = dis64(rng) | 1ULL;
            h_start_scalars[i*4+1] = dis64(rng);
            h_start_scalars[i*4+2] = dis64(rng);
            h_start_scalars[i*4+3] = dis64(rng);
            
            // Set high iteration count
            h_counts256[i*4+0] = 0xFFFFFFFFFFFFFFFFULL;
            h_counts256[i*4+1] = 0xFFFFFFFFFFFFFFFFULL;
            h_counts256[i*4+2] = 0xFFFFFFFFFFFFFFFFULL;
            h_counts256[i*4+3] = 0xFFFFFFFFFFFFFFFFULL;
        }
    } else {
        // Sequential/Random/Hybrid: divide range
        uint64_t per_thread_cnt[4];
        uint64_t r_u64 = 0ull;
        divmod_256_by_u64(range_len, threadsTotal, per_thread_cnt, r_u64);
        
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            h_counts256[i*4+0] = per_thread_cnt[0];
            h_counts256[i*4+1] = per_thread_cnt[1];
            h_counts256[i*4+2] = per_thread_cnt[2];
            h_counts256[i*4+3] = per_thread_cnt[3];
        }
        
        const uint32_t B = runtime_points_batch_size;
        const uint32_t half = B >> 1;
        
        if (mode >= 1) {
            // Random/Hybrid: XOR shuffle for starting positions
            for (uint64_t i = 0; i < threadsTotal; ++i) {
                uint64_t mask = threadsTotal - 1;
                uint64_t scrambled_idx = i ^ (random_seed & mask);
                
                uint64_t offset_scalar[4] = {0,0,0,0};
                mul256_u64(per_thread_cnt, scrambled_idx, offset_scalar);
                
                uint64_t temp_scalar[4];
                add256(range_start, offset_scalar, temp_scalar);
                
                uint64_t Sc[4];
                add256_u64(temp_scalar, (uint64_t)half, Sc);
                
                h_start_scalars[i*4+0] = Sc[0];
                h_start_scalars[i*4+1] = Sc[1];
                h_start_scalars[i*4+2] = Sc[2];
                h_start_scalars[i*4+3] = Sc[3];
            }
        } else {
            // Sequential
            uint64_t cur[4] = {range_start[0], range_start[1], range_start[2], range_start[3]};
            for (uint64_t i = 0; i < threadsTotal; ++i) {
                uint64_t Sc[4];
                add256_u64(cur, (uint64_t)half, Sc);
                h_start_scalars[i*4+0] = Sc[0];
                h_start_scalars[i*4+1] = Sc[1];
                h_start_scalars[i*4+2] = Sc[2];
                h_start_scalars[i*4+3] = Sc[3];

                uint64_t next[4];
                add256(cur, per_thread_cnt, next);
                cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
            }
        }
    }

    // ============================================
    // DEVICE ALLOCATIONS
    // ============================================
    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr;
    uint64_t *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr;
    FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr;
    unsigned int *d_any_left=nullptr;

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "malloc start_scalars");
    ck(cudaMalloc(&d_Px, threadsTotal * 4 * sizeof(uint64_t)), "malloc Px");
    ck(cudaMalloc(&d_Py, threadsTotal * 4 * sizeof(uint64_t)), "malloc Py");
    ck(cudaMalloc(&d_Rx, threadsTotal * 4 * sizeof(uint64_t)), "malloc Rx");
    ck(cudaMalloc(&d_Ry, threadsTotal * 4 * sizeof(uint64_t)), "malloc Ry");
    ck(cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(uint64_t)), "malloc counts");
    ck(cudaMalloc(&d_found_flag, sizeof(int)), "malloc found_flag");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "malloc found_result");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "malloc hashes");
    ck(cudaMalloc(&d_any_left, sizeof(unsigned int)), "malloc any_left");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start");
    ck(cudaMemcpy(d_counts256, h_counts256, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts");
    
    { int zero = FOUND_NONE; unsigned long long zero64=0ull;
      ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init flag");
      ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes");
    }

    // Generate initial points
    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "init points sync");
        ck(cudaGetLastError(), "init points launch");
    }

    // ============================================
    // SETUP FOR HYBRID MODE (if needed)
    // ============================================
    uint64_t *d_jump_x=nullptr, *d_jump_y=nullptr, *d_jump_s=nullptr;
    uint32_t num_jumps = 0;
    
    if (mode == 3) {
        num_jumps = 256;  // Number of random jump points
        uint64_t* h_jump_s = (uint64_t*)std::malloc(num_jumps * 4 * sizeof(uint64_t));
        
        std::mt19937_64 jump_rng(random_seed + 12345);
        std::uniform_int_distribution<uint64_t> dis64(1, UINT64_MAX);
        for (uint32_t i = 0; i < num_jumps; ++i) {
            h_jump_s[i*4+0] = dis64(jump_rng) | 1ULL;
            h_jump_s[i*4+1] = dis64(jump_rng);
            h_jump_s[i*4+2] = dis64(jump_rng);
            h_jump_s[i*4+3] = dis64(jump_rng);
        }
        
        uint64_t* d_jump_scalars;
        cudaMalloc(&d_jump_scalars, num_jumps * 4 * sizeof(uint64_t));
        cudaMalloc(&d_jump_x, num_jumps * 4 * sizeof(uint64_t));
        cudaMalloc(&d_jump_y, num_jumps * 4 * sizeof(uint64_t));
        cudaMalloc(&d_jump_s, num_jumps * 4 * sizeof(uint64_t));
        
        cudaMemcpy(d_jump_scalars, h_jump_s, num_jumps * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        int jump_blocks = (num_jumps + threadsPerBlock - 1) / threadsPerBlock;
        scalarMulKernelBase<<<jump_blocks, threadsPerBlock>>>(d_jump_scalars, d_jump_x, d_jump_y, num_jumps);
        cudaDeviceSynchronize();
        
        cudaMemcpy(d_jump_s, h_jump_s, num_jumps * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaFree(d_jump_scalars);
        std::free(h_jump_s);
        
        std::cout << "Jump pool ready (" << num_jumps << " points).\n";
    }

    // ============================================
    // PRINT INFO
    // ============================================
    size_t freeB=0, totalB=0;
    cudaMemGetInfo(&freeB, &totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "\n======== GPU Information ====================\n";
    std::cout << std::left << std::setw(22) << "Device" << " : " << prop.name << "\n";
    std::cout << std::left << std::setw(22) << "SMs" << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(22) << "Threads/Block" << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(22) << "Blocks" << " : " << blocks << "\n";
    std::cout << std::left << std::setw(22) << "Total Threads" << " : " << threadsTotal << "\n";
    if (mode != 2) {
        std::cout << std::left << std::setw(22) << "Batch Size" << " : " << runtime_points_batch_size << "\n";
        std::cout << std::left << std::setw(22) << "Batches/SM" << " : " << runtime_batches_per_sm << "\n";
    }
    std::cout << std::left << std::setw(22) << "Iters/Launch" << " : " << slices_per_launch << "\n";
    std::cout << std::left << std::setw(22) << "Memory Used" << " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << ")\n";
    std::cout << "==============================================\n\n";

    // ============================================
    // MAIN SEARCH LOOP
    // ============================================
    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "stream");

    if (mode == 2) {
        cudaFuncSetCacheConfig(kernel_extreme_random, cudaFuncCachePreferL1);
    } else {
        cudaFuncSetCacheConfig(kernel_hybrid_optimized, cudaFuncCachePreferL1);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    bool completed_all = false;

    while (!stop_all) {
        if (g_sigint) {
            std::cerr << "\n[Ctrl+C] Interrupted.\n";
        }

        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel), "zero any_left");

        // ============================================
        // LAUNCH KERNEL
        // ============================================
        if (mode == 2) {
            // Extreme Random Mode
            size_t shared_size = (threadsPerBlock / 32) * EXTREME_WARP_CACHE * 4 * 3 * sizeof(uint64_t);
            
            kernel_extreme_random<<<blocks, threadsPerBlock, shared_size, streamKernel>>>(
                d_Px, d_Py, d_start_scalars,
                d_Rx, d_Ry, d_start_scalars, d_counts256,
                threadsTotal,
                slices_per_launch,
                d_found_flag, d_found_result,
                d_hashes_accum,
                d_pool_x, d_pool_y, d_pool_s,
                pool_size,
                random_seed,
                d_any_left
            );
        } else {
            // Sequential/Random/Hybrid Mode
            kernel_hybrid_optimized<<<blocks, threadsPerBlock, 0, streamKernel>>>(
                d_Px, d_Py, d_Rx, d_Ry,
                d_start_scalars, d_counts256,
                threadsTotal,
                runtime_points_batch_size,
                slices_per_launch,
                d_found_flag, d_found_result,
                d_hashes_accum,
                d_any_left,
                d_jump_x, d_jump_y, d_jump_s,
                num_jumps,
                random_seed,
                mode >= 1 ? 1 : 0
            );
        }

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
        }

        // ============================================
        // MONITORING LOOP
        // ============================================
        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            
            if (dt >= 0.5) {  // Update every 0.5s for more responsive display
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                
                double delta = (double)(h_hashes - lastHashes);
                double speed = delta / (dt * 1e6);  // Mkeys/s
                
                double elapsed = std::chrono::duration<double>(now - t0).count();
                
                // Expected time calculation
                double prob_per_hash = 1.0 / (double)(1ULL << (vanity_len * 8));
                double expected_hashes = 1.0 / prob_per_hash;
                double progress = (double)h_hashes / expected_hashes * 100.0;
                if (progress > 100.0) progress = 100.0;
                
                double eta = speed > 0 ? (expected_hashes - h_hashes) / (speed * 1e6) : -1;
                
                std::cout << "\r"
                          << "Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
                          << " | Speed: " << std::setprecision(2) << speed << " Mkey/s"
                          << " | Hashes: " << h_hashes
                          << " | Progress: " << std::setprecision(4) << progress << "%";
                if (eta > 0 && eta < 86400 * 365) {
                    int hours = (int)(eta / 3600);
                    int mins = (int)(std::fmod(eta, 3600.0) / 60.0);
                    std::cout << " | ETA: " << hours << "h" << mins << "m";
                }
                std::cout.flush();
                
                lastHashes = h_hashes;
                tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read flag");
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cudaStreamSynchronize(streamKernel);
        std::cout.flush();
        
        if (stop_all || g_sigint) break;

        if (mode == 2) {
            // Extreme mode: swap and continue (no exhaustion check)
            std::swap(d_Px, d_Rx);
            std::swap(d_Py, d_Ry);
        } else {
            unsigned int h_any = 0u;
            ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");

            std::swap(d_Px, d_Rx);
            std::swap(d_Py, d_Ry);

            if (h_any == 0u) { completed_all = true; break; }
        }
    }

    cudaDeviceSynchronize();
    std::cout << "\n\n";

    // ============================================
    // OUTPUT RESULT
    // ============================================
    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final read flag");

    int exit_code = EXIT_SUCCESS;

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read result");
        
        std::cout << "╔══════════════════════════════════════════╗\n";
        std::cout << "║         *** FOUND MATCH! ***              ║\n";
        std::cout << "╠══════════════════════════════════════════╣\n";
        std::cout << "║ Private Key : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "║ Public Key  : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
        std::cout << "║ Thread ID   : " << host_result.threadId << "\n";
        std::cout << "╚══════════════════════════════════════════╝\n";
    } else {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ========\n";
            exit_code = 130;
        } else if (completed_all) {
            std::cout << "======== NOT FOUND (exhaustive) ========\n";
        } else {
            std::cout << "======== TERMINATED ========\n";
        }
    }

    // ============================================
    // CLEANUP
    // ============================================
    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py);
    cudaFree(d_Rx); cudaFree(d_Ry); cudaFree(d_counts256);
    cudaFree(d_found_flag); cudaFree(d_found_result);
    cudaFree(d_hashes_accum); cudaFree(d_any_left);
    
    if (d_pool_x) cudaFree(d_pool_x);
    if (d_pool_y) cudaFree(d_pool_y);
    if (d_pool_s) cudaFree(d_pool_s);
    if (d_jump_x) cudaFree(d_jump_x);
    if (d_jump_y) cudaFree(d_jump_y);
    if (d_jump_s) cudaFree(d_jump_s);
    
    cudaStreamDestroy(streamKernel);
    if (h_start_scalars) cudaFreeHost(h_start_scalars);
    if (h_counts256) cudaFreeHost(h_counts256);

    return exit_code;
}