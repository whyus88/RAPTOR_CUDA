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

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Shotgun: Table titik di jarak STRIDE
__constant__ uint64_t c_ShotgunX[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_ShotgunY[(MAX_BATCH_SIZE/2) * 4];
// Shotgun: Jump point = (half * STRIDE) * G
__constant__ uint64_t c_JumpPointX[4];
__constant__ uint64_t c_JumpPointY[4];
__constant__ int c_vanity_len;
__constant__ uint64_t c_RangeStart[4];
__constant__ uint64_t c_RangeEnd[4];
// Shotgun stride sebagai scalar
__constant__ uint64_t c_Stride[4];
// Shotgun: half * stride untuk update scalar
__constant__ uint64_t c_ScalarJump[4];

// Ultra-fast prefix check constants
__constant__ uint64_t c_prefix_64;  // Untuk 8-byte vanity

__launch_bounds__(256, 4)  // Tingkatkan occupancy
__global__ void kernel_shotgun_grasshopper(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_shots_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint64_t prefix64 = c_prefix_64;
    const int vanity_len = c_vanity_len;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 32768u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    // Load state
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

    // Local copies untuk speed
    uint64_t range_end[4], scalar_jump[4], stride[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        range_end[i] = c_RangeEnd[i];
        scalar_jump[i] = c_ScalarJump[i];
        stride[i] = c_Stride[i];
    }

    uint32_t shots_done = 0;

    // =====================================================
    // SHOTGUN MAIN LOOP
    // Setiap iterasi: check B point yang tersebar (sparse)
    // Lalu lompat jauh oleh scalar_jump
    // =====================================================
    while (shots_done < max_shots_per_launch) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); break; }
        
        // Cek apakah masih ada ruang untuk minimal 1 shot
        // Scalar akhir setelah shot ini: S + half * stride
        uint64_t max_scalar[4];
        bool overflow = false;
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __uint128_t res = (__uint128_t)S[i] + scalar_jump[i] + carry;
            max_scalar[i] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);
        }
        if (carry || cmp256_gt(max_scalar, range_end)) {
            break; // Out of range
        }

        // --- SHOTGUN HASH CHECK: Base point (offset 0) ---
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            // ULTRA-FAST: 1 instruction untuk 99.9999% rejection
            if (*(const uint64_t*)h20 == prefix64) {
                // Additional check for 8+ bytes
                bool match = true;
                #pragma unroll
                for (int k = 8; k < vanity_len && k < 20; ++k) {
                    if (h20[k] != c_target_hash160[k]) { match = false; break; }
                }
                if (match) {
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
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }
        }

        // --- SHOTGUN BATCH INVERSION ---
        // Compute product of (ShotgunX[i] - x1) for i = half-1 down to 0
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        #pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_JumpPointX[j];
        ModSub256(acc, acc, x1);
        #pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
            #pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_ShotgunX[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
            #pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
        #pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_ShotgunX[0*4 + j];
        ModSub256(d0, d0, x1);
        #pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        // --- SHOTGUN LOOP: Check sparse points +Shotgun[i] dan -Shotgun[i] ---
        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // BLOK +Shotgun[i]
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                #pragma unroll
                for (int j=0;j<4;++j) { 
                    px_i[j]=c_ShotgunX[(size_t)i*4+j]; 
                    py_i[j]=c_ShotgunY[(size_t)i*4+j]; 
                }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);     
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3); 
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; 
                getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                if (*(const uint64_t*)h20 == prefix64) {
                    bool match = true;
                    #pragma unroll
                    for (int k = 8; k < vanity_len && k < 20; ++k) {
                        if (h20[k] != c_target_hash160[k]) { match = false; break; }
                    }
                    if (match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; 
                            #pragma unroll
                            for (int k=0;k<4;++k) fs[k]=S[k];
                            // Add (i+1) * stride
                            uint64_t add_scalar[4];
                            uint64_t addv = (uint64_t)(i + 1);
                            #pragma unroll
                            for (int k=0;k<4;++k) {
                                __uint128_t res = (__uint128_t)stride[k] * addv;
                                add_scalar[k] = (uint64_t)res;
                            }
                            carry = 0;
                            #pragma unroll
                            for (int k=0;k<4;++k) {
                                __uint128_t res = (__uint128_t)fs[k] + add_scalar[k] + carry;
                                fs[k] = (uint64_t)res;
                                carry = (uint64_t)(res >> 64);
                            }
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4], t[4]; 
                            ModSub256(t, x1, px3); 
                            _ModMult(y3, t, lam); 
                            ModSub256(y3, y3, y1);
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

            // BLOK -Shotgun[i]
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                #pragma unroll
                for (int j=0;j<4;++j) { 
                    px_i[j]=c_ShotgunX[(size_t)i*4+j]; 
                    py_i[j]=c_ShotgunY[(size_t)i*4+j]; 
                }
                ModNeg256(py_i, py_i); 

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; 
                getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                if (*(const uint64_t*)h20 == prefix64) {
                    bool match = true;
                    #pragma unroll
                    for (int k = 8; k < vanity_len && k < 20; ++k) {
                        if (h20[k] != c_target_hash160[k]) { match = false; break; }
                    }
                    if (match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; 
                            #pragma unroll
                            for (int k=0;k<4;++k) fs[k]=S[k];
                            // Sub (i+1) * stride (with wrap for negative)
                            uint64_t sub_scalar[4];
                            uint64_t subv = (uint64_t)(i + 1);
                            #pragma unroll
                            for (int k=0;k<4;++k) {
                                __uint128_t res = (__uint128_t)stride[k] * subv;
                                sub_scalar[k] = (uint64_t)res;
                            }
                            uint64_t borrow = 0;
                            #pragma unroll
                            for (int k=0;k<4;++k) {
                                uint64_t val = fs[k];
                                uint64_t sub = sub_scalar[k] + borrow;
                                borrow = (val < sub) ? 1ull : 0ull;
                                fs[k] = val - sub;
                            }
                            // Add range_len to handle negative
                            #pragma unroll
                            for (int k=0;k<4;++k) {
                                __uint128_t res = (__uint128_t)fs[k] + c_RangeLen[k] + borrow;
                                fs[k] = (uint64_t)res;
                                borrow = (uint64_t)(res >> 64);
                            }
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4], t[4]; 
                            ModSub256(t, x1, px3); 
                            _ModMult(y3, t, lam); 
                            ModSub256(y3, y3, y1);
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

            // Update inverse untuk iterasi berikutnya
            uint64_t sx_i[4];
            #pragma unroll
            for (int j=0;j<4;++j) sx_i[j] = c_ShotgunX[(size_t)i*4 + j];
            ModSub256(sx_i, sx_i, x1);
            _ModMult(inverse, inverse, sx_i);
        }

        // --- SHOTGUN LAST POINT: -Shotgun[half-1] ---
        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
            #pragma unroll
            for (int j=0;j<4;++j) { 
                px_i[j]=c_ShotgunX[(size_t)i*4+j]; 
                py_i[j]=c_ShotgunY[(size_t)i*4+j]; 
            }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            uint8_t h20[20]; 
            getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            if (*(const uint64_t*)h20 == prefix64) {
                bool match = true;
                #pragma unroll
                for (int k = 8; k < vanity_len && k < 20; ++k) {
                    if (h20[k] != c_target_hash160[k]) { match = false; break; }
                }
                if (match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; 
                        #pragma unroll
                        for (int k=0;k<4;++k) fs[k]=S[k];
                        // Sub half * stride
                        uint64_t borrow = 0;
                        #pragma unroll
                        for (int k=0;k<4;++k) {
                            uint64_t val = fs[k];
                            uint64_t sub = scalar_jump[k] + borrow;
                            borrow = (val < sub) ? 1ull : 0ull;
                            fs[k] = val - sub;
                        }
                        #pragma unroll
                        for (int k=0;k<4;++k) {
                            __uint128_t res = (__uint128_t)fs[k] + c_RangeLen[k] + borrow;
                            fs[k] = (uint64_t)res;
                            borrow = (uint64_t)(res >> 64);
                        }
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4], t[4]; 
                        ModSub256(t, x1, px3); 
                        _ModMult(y3, t, lam); 
                        ModSub256(y3, y3, y1);
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

        // =====================================================
        // SHOTGUN JUMP: P = P + JumpPoint, S = S + scalar_jump
        // =====================================================
        {
            uint64_t lam[4], s[4];
            uint64_t Jy_minus_y1[4];
            #pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_JumpPointY[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            // Reuse inverse dari akhir loop (sudah = 1/(JumpPointX - x1))
            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x1, lam);
            ModSub256(x1, x1, x1); // BUG FIX: seharusnya menggunakan tmp
            // Corrected:
            uint64_t x_old[4];
            #pragma unroll
            for(int k=0;k<4;++k) x_old[k] = x1[k];
            _ModSqr(x1, lam);
            ModSub256(x1, x1, x_old);
            uint64_t Jx_local[4]; 
            #pragma unroll
            for (int j=0;j<4;++j) Jx_local[j]=c_JumpPointX[j];
            ModSub256(x1, x1, Jx_local);

            ModSub256(s, x_old, x1);
            _ModMult(y1, s, lam);
            ModSub256(y1, y1, y_old);
        }

        // Update scalar
        carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __uint128_t res = (__uint128_t)S[i] + scalar_jump[i] + carry;
            S[i] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);
        }

        // Update remaining (approximate - tidak perlu exact untuk shotgun)
        sub256_u64_inplace(rem, (uint64_t)(half * 2)); // Approx
        
        ++shots_done;
    }

    // Store final state
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    
    // Mark jika masih bisa lanjut
    if (cmp256_le(S, range_end)) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// Helper untuk membandingkan 256-bit
__device__ bool cmp256_gt(const uint64_t a[4], const uint64_t b[4]) {
    if (a[3] != b[3]) return a[3] > b[3];
    if (a[2] != b[2]) return a[2] > b[2];
    if (a[1] != b[1]) return a[1] > b[1];
    return a[0] > b[0];
}

__device__ bool cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    return !cmp256_gt(a, b);
}

// Deklarasi eksternal
extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_batch_size = 128;
    uint32_t runtime_batches_per_sm = 8;
    uint32_t shots_per_launch = 256;  // Shotgun: jumlah "tembakan" per launch
    uint32_t stride_bits = 16;        // Shotgun stride = 2^stride_bits

    // Parser args
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
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\"\n";
                return EXIT_FAILURE;
            }
            runtime_batch_size = a;
            runtime_batches_per_sm = b;
        }
        else if (arg == "--shots" && i + 1 < argc) {
            shots_per_launch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        }
        else if (arg == "--stride-bits" && i + 1 < argc) {
            stride_bits = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
            if (stride_bits < 8 || stride_bits > 24) {
                std::cerr << "Error: --stride-bits must be 8-24\n";
                return EXIT_FAILURE;
            }
        }
    }

    if (range_hex.empty() || vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start:end> --vanity-hash160 <hex>"
                  << " [--grid A,B] [--shots N] [--stride-bits N]\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { 
        std::cerr << "Error: range format must be start:end\n"; 
        return EXIT_FAILURE; 
    }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; 
        return EXIT_FAILURE;
    }

    // Parse Vanity
    uint8_t target_hash160[20];
    memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);

    if (vanity_len < 8) {
        std::cerr << "Warning: Shotgun mode optimized for 8+ byte vanity\n";
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_batch_size) || (runtime_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even power of two\n";
        return EXIT_FAILURE;
    }
    if (runtime_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size > " << MAX_BATCH_SIZE << "\n";
        return EXIT_FAILURE;
    }

    // Range length (untuk statistik)
    uint64_t range_len[4]; 
    sub256(range_end, range_start, range_len); 
    add256_u64(range_len, 1ull, range_len);

    // Calculate range bits
    auto popcount256 = [](const uint64_t a[4])->int {
        int cnt = 0;
        for (int i = 0; i < 4; ++i) cnt += __builtin_popcountll(a[i]);
        return cnt;
    };
    auto clz256 = [](const uint64_t a[4])->int {
        for (int i = 3; i >= 0; --i) {
            if (a[i] != 0) return __builtin_clzll(a[i]) + (3 - i) * 64;
        }
        return 256;
    };
    int range_bits = 256 - clz256(range_len);
    
    // Auto-adjust stride if needed
    if (stride_bits > range_bits - 8) {
        stride_bits = range_bits - 8;
        std::cout << "Info: Auto-adjusted stride-bits to " << stride_bits << "\n";
    }

    // Setup GPU
    int device=0; 
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; 
        return EXIT_FAILURE;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) 
        threadsPerBlock = prop.maxThreadsPerBlock;

    const uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    uint64_t usableMem = (totalGlobalMem > 64ull*1024*1024) ? 
                         (totalGlobalMem - 64ull*1024*1024) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    // Threads total: untuk shotgun, kita bisa gunakan lebih banyak
    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * 
                         (uint64_t)runtime_batches_per_sm * 
                         (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    uint64_t threadsTotal = std::min(maxThreadsByMem, userUpper);
    threadsTotal = threadsTotal - (threadsTotal % threadsPerBlock);
    if (threadsTotal == 0) threadsTotal = threadsPerBlock;
    
    int blocks = (int)(threadsTotal / threadsPerBlock);

    // ==========================================
    // SHOTGUN PARAMETERS
    // ==========================================
    const uint32_t B = runtime_batch_size;
    const uint32_t half = B >> 1;
    
    // Stride = 2^stride_bits
    uint64_t stride[4] = {0};
    stride[stride_bits / 64] = 1ULL << (stride_bits % 64);
    
    // Scalar jump = half * stride (untuk lompat antar shot)
    uint64_t scalar_jump[4] = {0};
    {
        uint64_t half_val = half;
        int shift = stride_bits;
        while (half_val && shift < 256) {
            if (half_val & 1) {
                // Add stride << shift
                uint64_t carry = 0;
                int word_shift = shift / 64;
                int bit_shift = shift % 64;
                for (int k = 0; k < 4; ++k) {
                    if (k + word_shift < 4) {
                        __uint128_t res = (__uint128_t)scalar_jump[k + word_shift] + 
                                         ((__uint128_t)stride[k] << bit_shift) + carry;
                        scalar_jump[k + word_shift] = (uint64_t)res;
                        carry = (uint64_t)(res >> 64);
                    } else if (carry) {
                        break;
                    }
                }
            }
            half_val >>= 1;
            shift++;
        }
    }

    std::cout << "Info: SHOTGUN MODE - Stride=2^" << stride_bits 
              << ", Shots/launch=" << shots_per_launch << "\n";

    // Alokasi Memori
    uint64_t* h_start_scalars = nullptr;
    uint64_t* h_counts256 = nullptr;
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), 
                  cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_counts256, threadsTotal * 4 * sizeof(uint64_t), 
                  cudaHostAllocWriteCombined | cudaHostAllocMapped);

    // Initialize: Random starts dengan jaminan dalam range
    std::random_device rd;
    std::mt19937_64 gen(rd());
    
    // Hitung max start scalar (range_end - scalar_jump agar bisa minimal 1 shot)
    uint64_t max_start[4];
    {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            uint64_t sub = scalar_jump[i] + borrow;
            borrow = (range_end[i] < sub) ? 1 : 0;
            max_start[i] = range_end[i] - sub;
        }
    }

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        // Random scalar dalam [range_start, max_start]
        uint64_t rand_val[4];
        for (int k = 0; k < 4; ++k) rand_val[k] = gen();
        
        // Modulo dengan range (approximate - ok untuk shotgun)
        uint64_t range_span[4];
        sub256(max_start, range_start, range_span);
        add256_u64(range_span, 1ull, range_span);
        
        // Simple modulo (good enough for random)
        for (int k = 0; k < 4; ++k) rand_val[k] %= range_span[k];
        
        add256(range_start, rand_val, &h_start_scalars[i*4]);
        
        // Counts: approximate, tidak critical untuk shotgun
        uint64_t remaining[4];
        sub256(range_end, &h_start_scalars[i*4], remaining);
        add256_u64(remaining, 1ull, remaining);
        for (int k = 0; k < 4; ++k) h_counts256[i*4+k] = remaining[k];
    }

    // Copy Constants to Device
    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));
    cudaMemcpyToSymbol(c_RangeStart, range_start, 4*sizeof(uint64_t));
    cudaMemcpyToSymbol(c_RangeEnd, range_end, 4*sizeof(uint64_t));
    cudaMemcpyToSymbol(c_RangeLen, range_len, 4*sizeof(uint64_t));
    cudaMemcpyToSymbol(c_Stride, stride, 4*sizeof(uint64_t));
    cudaMemcpyToSymbol(c_ScalarJump, scalar_jump, 4*sizeof(uint64_t));

    // Ultra-fast prefix for 8-byte vanity
    uint64_t prefix64 = 0;
    if (vanity_len >= 8) {
        memcpy(&prefix64, target_hash160, 8);
    } else if (vanity_len >= 4) {
        memcpy(&prefix64, target_hash160, vanity_len);
    }
    cudaMemcpyToSymbol(c_prefix_64, &prefix64, sizeof(prefix64));

    // Device allocations
    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr;
    uint64_t *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr; 
    FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr;
    unsigned int *d_any_left=nullptr;

    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "malloc start");
    ck(cudaMalloc(&d_Px, threadsTotal * 4 * sizeof(uint64_t)), "malloc Px");
    ck(cudaMalloc(&d_Py, threadsTotal * 4 * sizeof(uint64_t)), "malloc Py");
    ck(cudaMalloc(&d_Rx, threadsTotal * 4 * sizeof(uint64_t)), "malloc Rx");
    ck(cudaMalloc(&d_Ry, threadsTotal * 4 * sizeof(uint64_t)), "malloc Ry");
    ck(cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(uint64_t)), "malloc counts");
    ck(cudaMalloc(&d_found_flag, sizeof(int)), "malloc flag");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "malloc result");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "malloc hashes");
    ck(cudaMalloc(&d_any_left, sizeof(unsigned int)), "malloc any_left");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars, 
                  threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start");
    ck(cudaMemcpy(d_counts256, h_counts256, 
                  threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts");
    { 
        int zero = FOUND_NONE; 
        unsigned long long zero64=0ull;
        ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init flag");
        ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), 
                      cudaMemcpyHostToDevice), "init hashes"); 
    }

    // ==========================================
    // PRECOMPUTE SHOTGUN TABLE
    // ShotgunX[i] = (i+1) * STRIDE * G
    // ==========================================
    std::cout << "Info: Precomputing Shotgun table...\n";
    {
        uint64_t* h_shotgun_scalars = (uint64_t*)std::malloc(half * 4 * sizeof(uint64_t));
        memset(h_shotgun_scalars, 0, half * 4 * sizeof(uint64_t));
        
        for (uint32_t i = 0; i < half; ++i) {
            // scalar = (i+1) * stride
            uint64_t mult = (uint64_t)(i + 1);
            int shift = stride_bits;
            while (mult) {
                if (mult & 1) {
                    uint64_t carry = 0;
                    int word_shift = shift / 64;
                    int bit_shift = shift % 64;
                    for (int k = 0; k < 4; ++k) {
                        if (k + word_shift < 4) {
                            __uint128_t res = (__uint128_t)h_shotgun_scalars[(size_t)i*4 + k + word_shift] + 
                                             ((__uint128_t)stride[k] << bit_shift) + carry;
                            h_shotgun_scalars[(size_t)i*4 + k + word_shift] = (uint64_t)res;
                            carry = (uint64_t)(res >> 64);
                        }
                    }
                }
                mult >>= 1;
                shift++;
            }
        }

        uint64_t *d_shotgun_scalars=nullptr, *d_Sx=nullptr, *d_Sy=nullptr;
        ck(cudaMalloc(&d_shotgun_scalars, half * 4 * sizeof(uint64_t)), "malloc shotgun_s");
        ck(cudaMalloc(&d_Sx, half * 4 * sizeof(uint64_t)), "malloc Sx");
        ck(cudaMalloc(&d_Sy, half * 4 * sizeof(uint64_t)), "malloc Sy");
        ck(cudaMemcpy(d_shotgun_scalars, h_shotgun_scalars, 
                      half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy shotgun_s");

        int blocks_scal = (half + threadsPerBlock - 1) / threadsPerBlock;
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(
            d_shotgun_scalars, d_Sx, d_Sy, half);
        ck(cudaDeviceSynchronize(), "shotgun table sync");
        ck(cudaGetLastError(), "shotgun table launch");

        uint64_t* h_Sx = (uint64_t*)std::malloc(half * 4 * sizeof(uint64_t));
        uint64_t* h_Sy = (uint64_t*)std::malloc(half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Sx, d_Sx, half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Sx");
        ck(cudaMemcpy(h_Sy, d_Sy, half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Sy");
        ck(cudaMemcpyToSymbol(c_ShotgunX, h_Sx, half * 4 * sizeof(uint64_t)), "ToSymbol ShotgunX");
        ck(cudaMemcpyToSymbol(c_ShotgunY, h_Sy, half * 4 * sizeof(uint64_t)), "ToSymbol ShotgunY");

        cudaFree(d_shotgun_scalars); cudaFree(d_Sx); cudaFree(d_Sy);
        std::free(h_shotgun_scalars); std::free(h_Sx); std::free(h_Sy);
    }

    // ==========================================
    // PRECOMPUTE JUMP POINT = scalar_jump * G
    // ==========================================
    std::cout << "Info: Precomputing Jump point...\n";
    {
        uint64_t h_jump_scalar[4];
        memcpy(h_jump_scalar, scalar_jump, 4 * sizeof(uint64_t));

        uint64_t *d_jump_s=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_jump_s, 4 * sizeof(uint64_t)), "malloc jump_s");
        ck(cudaMalloc(&d_Jx, 4 * sizeof(uint64_t)), "malloc Jx");
        ck(cudaMalloc(&d_Jy, 4 * sizeof(uint64_t)), "malloc Jy");
        ck(cudaMemcpy(d_jump_s, h_jump_scalar, 4 * sizeof(uint64_t), 
                      cudaMemcpyHostToDevice), "cpy jump_s");

        scalarMulKernelBase<<<1, 1>>>(d_jump_s, d_Jx, d_Jy, 1);
        ck(cudaDeviceSynchronize(), "jump point sync");
        ck(cudaGetLastError(), "jump point launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_JumpPointX, hJx, 4 * sizeof(uint64_t)), "ToSymbol JumpX");
        ck(cudaMemcpyToSymbol(c_JumpPointY, hJy, 4 * sizeof(uint64_t)), "ToSymbol JumpY");

        cudaFree(d_jump_s); cudaFree(d_Jx); cudaFree(d_Jy);
    }

    // ==========================================
    // PRECOMPUTE START POINTS
    // ==========================================
    std::cout << "Info: Precomputing start points...\n";
    {
        int blocks_scal = (threadsTotal + threadsPerBlock - 1) / threadsPerBlock;
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(
            d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "start points sync");
        ck(cudaGetLastError(), "start points launch");
    }

    // Info
    size_t freeB=0, totalB=0; 
    cudaMemGetInfo(&freeB, &totalB);

    std::cout << "\n======== SHOTGUN GRASSHOPPER =========================\n";
    std::cout << std::left << std::setw(22) << "Mode"          << " : SHOTGUN (Sparse Sampling)\n";
    std::cout << std::left << std::setw(22) << "Device"        << " : " << prop.name 
              << " (SM " << prop.multiProcessorCount << ")\n";
    std::cout << std::left << std::setw(22) << "ThreadsTotal"  << " : " << threadsTotal << "\n";
    std::cout << std::left << std::setw(22) << "Batch Size"    << " : " << B << "\n";
    std::cout << std::left << std::setw(22) << "Stride"        << " : 2^" << stride_bits 
              << " = " << (1ULL << stride_bits) << "\n";
    std::cout << std::left << std::setw(22) << "Shotgun Spread" << " : " << (half * (1ULL << stride_bits)) 
              << " per batch\n";
    std::cout << std::left << std::setw(22) << "Shots/Launch"  << " : " << shots_per_launch << "\n";
    std::cout << std::left << std::setw(22) << "Vanity Target" << " : " << vanity_hash_hex 
              << " (" << vanity_len << " bytes)\n";
    std::cout << std::left << std::setw(22) << "Range Bits"    << " : ~" << range_bits << "\n";
    std::cout << "\n======== FIRING SHOTS ================================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "stream");

    (void)cudaFuncSetCacheConfig(kernel_shotgun_grasshopper, cudaFuncCachePreferL1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;
    unsigned int launch_count = 0;

    bool stop_all = false;
    while (!stop_all) {
        if (g_sigint) {
            std::cerr << "\n[Ctrl+C] Interrupted...\n";
        }

        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), 
                          cudaMemcpyHostToDevice, streamKernel), "zero any_left");

        kernel_shotgun_grasshopper<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars, d_counts256,
            threadsTotal,
            B,
            shots_per_launch,
            d_found_flag, d_found_result,
            d_hashes_accum,
            d_any_left
        );
        
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
        }

        launch_count++;

        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            
            if (dt >= 0.5) {  // More frequent updates
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), 
                             cudaMemcpyDeviceToHost), "read hashes");
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                
                long double total_keys_ld = ld_from_u256(range_len);
                long double coverage = 0.0L;
                if (total_keys_ld > 0.0L) {
                    coverage = ((long double)h_hashes / total_keys_ld) * 100.0L;
                }
                
                std::cout << "\r[Shot #" << std::setw(4) << launch_count << "] "
                          << "Time: " << std::fixed << std::setprecision(1) << elapsed << "s | "
                          << "Speed: " << std::setprecision(2) << mkeys << " Mkeys/s | "
                          << "Hashes: " << h_hashes << " | "
                          << "Coverage: " << std::setprecision(6) << (double)coverage << "%   ";
                std::cout.flush();
                lastHashes = h_hashes; 
                tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), 
                         cudaMemcpyDeviceToHost), "read flag");
            if (host_found == FOUND_READY) { 
                stop_all = true; 
                break; 
            }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { 
                cudaGetLastError(); 
                stop_all = true; 
                break; 
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cudaStreamSynchronize(streamKernel);
        std::cout.flush();
        if (stop_all || g_sigint) break;

        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

        unsigned int h_any = 0u;
        ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), 
                     cudaMemcpyDeviceToHost), "read any_left");
        if (h_any == 0u) {
            std::cout << "\nInfo: All threads reached range boundary. Reinitializing...\n";
            // Re-randomize for another pass
            for (uint64_t i = 0; i < threadsTotal; ++i) {
                uint64_t rand_val[4];
                for (int k = 0; k < 4; ++k) rand_val[k] = gen();
                
                uint64_t range_span[4];
                sub256(max_start, range_start, range_span);
                add256_u64(range_span, 1ull, range_span);
                for (int k = 0; k < 4; ++k) rand_val[k] %= range_span[k];
                
                add256(range_start, rand_val, &h_start_scalars[i*4]);
            }
            ck(cudaMemcpy(d_start_scalars, h_start_scalars, 
                         threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "reinit start");
            
            // Recompute points
            int blocks_scal = (threadsTotal + threadsPerBlock - 1) / threadsPerBlock;
            scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(
                d_start_scalars, d_Px, d_Py, (int)threadsTotal);
            ck(cudaDeviceSynchronize(), "reinit sync");
            ck(cudaGetLastError(), "reinit launch");
        }
    }

    cudaDeviceSynchronize();
    std::cout << "\n\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), 
                 cudaMemcpyDeviceToHost), "final read flag");

    int exit_code = EXIT_SUCCESS;

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), 
                     cudaMemcpyDeviceToHost), "read result");
        std::cout << "\n========== BOOM! TARGET FOUND! =====================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
        std::cout << "Found by Thread: " << host_result.threadId << "\n";
    } else {
        if (g_sigint) {
            std::cout << "========== INTERRUPTED ==============================\n";
            exit_code = 130;
        } else {
            std::cout << "========== SEARCH EXHAUSTED ========================\n";
        }
    }

    // Cleanup
    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py);
    cudaFree(d_Rx); cudaFree(d_Ry); cudaFree(d_counts256);
    cudaFree(d_found_flag); cudaFree(d_found_result);
    cudaFree(d_hashes_accum); cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);
    cudaFreeHost(h_start_scalars);
    cudaFreeHost(h_counts256);

    return exit_code;
}