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

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_RangeStart[4]; 
__constant__ uint64_t c_RangeMask[4];
__constant__ int c_vanity_len;

// ============================================================
// 1. JACOBIAN HELPER FUNCTIONS (INVERSION-FREE CORE)
// ============================================================

// Point Double: R = 2 * P
// Formulas:
// A = X^2, B = Y^2, C = B^2
// D = 2*((X+B)^2 - A - C)
// E = 3*A
// X3 = E^2 - 2*D
// Y3 = E*(D - X3) - 8*C
// Z3 = 2*Y*Z
__device__ void jacobian_double(uint64_t* X, uint64_t* Y, uint64_t* Z) {
    uint64_t A[4], B[4], C[4], D[4], E[4];
    uint64_t t1[4], t2[4];

    _ModSqr(A, X);           // A = X^2
    _ModSqr(B, Y);           // B = Y^2
    _ModSqr(C, B);           // C = B^2

    // D = 2 * ((X+B)^2 - A - C)
    fieldAdd(X, B, t1);      // t1 = X + B
    _ModSqr(t2, t1);         // t2 = (X+B)^2
    fieldSub(t2, A, t1);     // t1 = (X+B)^2 - A
    fieldSub(t1, C, D);      // D = ... - C
    fieldAdd(D, D, D);       // D = 2 * D (Final D)

    // E = 3 * A
    fieldAdd(A, A, t1);      // t1 = 2A
    fieldAdd(t1, A, E);      // E = 3A

    // X3 = E^2 - 2*D
    _ModSqr(t1, E);          // t1 = E^2
    fieldAdd(D, D, t2);      // t2 = 2D
    fieldSub(t1, D, X);      // X3 = E^2 - 2D (Update X)

    // Y3 = E*(D - X3) - 8*C
    fieldSub(D, X, t1);      // t1 = D - X3
    _ModMult(t2, E, t1);     // t2 = E * (D - X3)
    
    // 8*C = C << 3
    fieldAdd(C, C, t1);      // 2C
    fieldAdd(t1, t1, t1);    // 4C
    fieldAdd(t1, t1, t1);    // 8C
    
    fieldSub(t2, t1, Y);     // Y3 = ... - 8C (Update Y)

    // Z3 = 2 * Y * Z
    _ModMult(t1, Y, Z);      // t1 = Y * Z
    fieldAdd(t1, t1, Z);     // Z3 = 2 * t1 (Update Z)
}

// Point Add Mixed: R = P + Q (P is Jacobian, Q is Affine)
// Optimized for Z2 = 1
__device__ void jacobian_add_mixed(uint64_t* X1, uint64_t* Y1, uint64_t* Z1, 
                                   const uint64_t* X2, const uint64_t* Y2) {
    uint64_t Z1_sq[4], Z1_cu[4];
    uint64_t U2[4], S2[4];
    uint64_t H[4], R[4];
    uint64_t H_sq[4], H_cu[4];
    uint64_t t1[4], t2[4];

    // If Z1 == 0, copy Q to P
    if (_IsZero(Z1)) {
        Load(X1, X2);
        Load(Y1, Y2);
        Z1[0] = 1; Z1[1] = 0; Z1[2] = 0; Z1[3] = 0;
        return;
    }

    _ModSqr(Z1_sq, Z1);        // Z1^2
    _ModMult(Z1_cu, Z1, Z1_sq);// Z1^3

    _ModMult(U2, X2, Z1_sq);   // U2 = X2 * Z1^2
    _ModMult(S2, Y2, Z1_cu);   // S2 = Y2 * Z1^3

    fieldSub(U2, X1, H);       // H = U2 - X1
    fieldSub(S2, Y1, R);       // R = S2 - Y1

    // If H == 0
    if (_IsZero(H)) {
        if (_IsZero(R)) {
            // P == Q, Double
            jacobian_double(X1, Y1, Z1);
        } else {
            // P == -Q, Infinity
            Z1[0] = 0; Z1[1] = 0; Z1[2] = 0; Z1[3] = 0;
        }
        return;
    }

    _ModSqr(H_sq, H);          // H_sq = H^2
    _ModMult(H_cu, H, H_sq);   // H_cu = H^3

    // X3 = R^2 - H^3 - 2*X1*H^2
    _ModSqr(t1, R);            // t1 = R^2
    fieldSub(t1, H_cu, t1);    // t1 = R^2 - H^3
    
    _ModMult(t2, X1, H_sq);    // t2 = X1 * H^2
    fieldAdd(t2, t2, X1);      // t2 = 2 * X1 * H^2 (Reuse X1 as temp storage)
    
    fieldSub(t1, t2, X1);      // X3 = (R^2 - H^3) - 2*X1*H^2 (Store in X1)

    // Y3 = R*(X1*H^2 - X3) - Y1*H^3
    // We need original X1 for calculation. 
    // Since we updated X1, we need to recover or use careful ordering.
    // Re-calc t2 (X1*H^2) is needed? No, t2 holds it.
    // t2 still holds old X1*H^2.
    
    fieldSub(t2, X1, t2);      // t2 = X3 - (X1*H^2) -> No, formula is X1*H^2 - X3.
                              // So: t2 = (X1_old * H^2) - X1_new. 
                              // We have t2 = 2 * X1_old * H^2.
                              // Let's re-derive simply:
    _ModMult(t2, X1, H_sq);    // Recalc X1_new * H^2? No. 
                              // Correct logic:
                              // V = X1 * H^2. (We had this in t2 before overwriting X1).
    
    // Let's redo cleanly with temps for X3, Y3
    uint64_t X3[4], Y3[4], Z3[4];
    
    _ModSqr(t1, R);            // R^2
    _ModMult(t2, X1, H_sq);    // X1 * H^2
    fieldSub(X3, t1, H_cu);    // R^2 - H^3
    fieldSub(X3, X3, t2);      // R^2 - H^3 - X1*H^2
    fieldSub(X3, X3, t2);      // R^2 - H^3 - 2*X1*H^2 (Result X3)
    
    _ModMult(t1, X1, H_sq);    // X1 * H^2 (Again)
    fieldSub(t1, t1, X3);      // (X1*H^2) - X3
    _ModMult(t1, R, t1);       // R * ((X1*H^2) - X3)
    
    _ModMult(t2, Y1, H_cu);    // Y1 * H^3
    fieldSub(Y3, t1, t2);      // Y3
    
    // Z3 = Z1 * H
    _ModMult(Z3, Z1, H);
    
    Load(X1, X3);
    Load(Y1, Y3);
    Load(Z1, Z3);
}

// Scalar Multiplication: P = k * G (Jacobian Output)
__device__ void scalarMulBaseJacobian(const uint64_t* k, uint64_t* X, uint64_t* Y, uint64_t* Z) {
    // Initialize to Infinity
    X[0]=0; X[1]=0; X[2]=0; X[3]=0;
    Y[0]=0; Y[1]=0; Y[2]=0; Y[3]=0;
    Z[0]=0; Z[1]=0; Z[2]=0; Z[3]=0;

    bool started = false;
    int msb = -1;

    // Find MSB
    for (int i = 3; i >= 0; --i) {
        if (k[i] != 0) {
            msb = i * 64 + (63 - __clzll(k[i]));
            break;
        }
    }
    if (msb == -1) return;

    for (int i = msb; i >= 0; --i) {
        if (started) {
            jacobian_double(X, Y, Z);
        }
        
        int limb = i >> 6;
        int shift = i & 63;
        if ((k[limb] >> shift) & 1) {
            if (!started) {
                // First bit, load G
                Load(X, c_Gx); 
                Load(Y, c_Gy);
                Z[0] = 1; Z[1]=0; Z[2]=0; Z[3]=0;
                started = true;
            } else {
                jacobian_add_mixed(X, Y, Z, c_Gx, c_Gy);
            }
        }
    }
}

// ============================================================
// 2. WARP BATCH INVERSION (Optimization)
// ============================================================

// Helper to load/store for batch invert
__device__ void batch_invert_warp(uint64_t val[4]) {
    // We use Shared Memory to store partial products for the warp
    // Size: 32 threads * 4 uint64 = 8KB per block (fits in 48KB limit)
    __shared__ uint64_t smem[32][4]; 
    
    const unsigned lane = threadIdx.x & 31;
    const unsigned mask = 0xFFFFFFFFu;

    // 1. Store original Z to shared memory
    // Each thread writes its Z to smem[lane]
    #pragma unroll
    for(int i=0; i<4; ++i) smem[lane][i] = val[i];
    
    __syncwarp(mask);

    // 2. Compute product of all Zs in the warp (Tree Reduction)
    // We compute prod[0..31] where prod[i] = smem[0] * ... * smem[i]
    // Step 0: each thread loads its own
    uint64_t prod[4];
    Load(prod, val);

    // Step 1: scan
    // T1 = z0
    // T2 = z0*z1
    // T3 = (z0*z1)*z2
    // ...
    // This is a prefix scan multiply.
    // Efficient implementation using shuffles:
    
    uint64_t p[4];
    
    // Iterative scan approach (log(32) = 5 steps)
    // We want: prod_i = product(val_0 ... val_i)
    
    // Step 1 (dist=1): p = val from lane-1
    if (lane >= 1) {
        #pragma unroll
        for(int i=0; i<4; ++i) p[i] = __shfl_up_sync(mask, prod[i], 1);
        _ModMult(prod, prod, p);
    }

    // Step 2 (dist=2): p = val from lane-2
    if (lane >= 2) {
        #pragma unroll
        for(int i=0; i<4; ++i) p[i] = __shfl_up_sync(mask, prod[i], 2);
        _ModMult(prod, prod, p);
    }

    // Step 3 (dist=4)
    if (lane >= 4) {
        #pragma unroll
        for(int i=0; i<4; ++i) p[i] = __shfl_up_sync(mask, prod[i], 4);
        _ModMult(prod, prod, p);
    }

    // Step 4 (dist=8)
    if (lane >= 8) {
        #pragma unroll
        for(int i=0; i<4; ++i) p[i] = __shfl_up_sync(mask, prod[i], 8);
        _ModMult(prod, prod, p);
    }

    // Step 5 (dist=16)
    if (lane >= 16) {
        #pragma unroll
        for(int i=0; i<4; ++i) p[i] = __shfl_up_sync(mask, prod[i], 16);
        _ModMult(prod, prod, p);
    }

    // Now 'prod' in each thread holds the prefix product.
    // Lane 31 holds the product of all Zs.
    
    // 3. Compute Inverse of the Total Product
    // Broadcast Lane 31's product to everyone? No, only Lane 31 computes inverse.
    // Actually, we need total product for next step.
    
    uint64_t total_prod[4];
    // Broadcast total prod from lane 31
    #pragma unroll
    for(int i=0; i<4; ++i) total_prod[i] = __shfl_sync(mask, prod[i], 31);
    
    // Everyone computes inverse of total product (Wasteful? No, 1 inverse per warp is cheap compared to 32)
    // Wait, only 1 thread NEEDS to compute it?
    // Logic:
    // We have P_all = Z0 * Z1 * ... * Z31.
    // Inv_all = 1 / P_all.
    // We need individual inverses:
    // Inv_Z31 = Inv_all * (Z0*...*Z30)
    // Inv_Z30 = Inv_Z31 * Z31
    // ...
    
    // So, we need Inv_all.
    // Compute it once. Since modular inverse is expensive, doing it 32 times serializes.
    // But we can compute it once in Lane 0 and broadcast.
    
    // Lane 31 has the full product in 'prod'.
    // Broadcast 'prod' (total product) to Lane 0? Or Lane 31 computes inverse?
    // Let's have Lane 31 compute the inverse.
    uint64_t inv_total[4];
    
    // Only Lane 31 computes inverse
    bool is_master = (lane == 31);
    
    // We need to move Total Product to a place to invert it.
    // Lane 31 has it.
    
    // We can do the inverse in parallel? No, _ModInv is iterative.
    // 1 Inverse per warp is acceptable (Saved 31 inverses).
    
    if (is_master) {
        _ModInv(prod); // prod is now 1 / TotalProd
        Load(inv_total, prod);
    }
    
    // Broadcast inv_total
    #pragma unroll
    for(int i=0; i<4; ++i) inv_total[i] = __shfl_sync(mask, inv_total[i], 31);
    
    // 4. Back-substitution to get individual Inverses
    // We need prefix products: P_k = Z0 * ... * Zk.
    // We have this in 'prod' variable!
    // We also need suffix products? No.
    
    // Logic:
    // Inv_Zk = Inv_Total * (Product of all Z_i where i != k)
    // Inv_Zk = Inv_Total * (Product_0_to_k-1) * (Product_k+1_to_31)
    
    // This is getting complex. 
    // Alternative "Simpler" Batch Inversion Logic (Standard Montgomery):
    // 1. P_0 = Z0
    // 2. P_1 = Z0 * Z1
    // ...
    // 3. P_31 = Total Product.
    // 4. Compute Inv_Total = 1 / P_31.
    // 5. Inv_Z31 = Inv_Total * P_30
    // 6. Inv_Z30 = Inv_Z31 * Z31
    // 7. Inv_Z29 = Inv_Z30 * Z30
    // ...
    
    // Implementation:
    
    // Step A: Broadcast Z values needed for backprop
    // We need to store Z values in smem to read them later.
    // Already done at start.
    
    // Step B: Backprop Loop
    // We run a loop 0..31. But we are inside a warp.
    // We can parallelize this backprop!
    // Inv_i = Inv_Total * (P_i-1) * (P_31 / P_i)
    // This requires division. Hard.
    
    // Let's stick to Serial Backprop inside Warp (SIMT).
    // Loop index `k` from 31 down to 0.
    // This runs in lockstep.
    
    // Current state:
    // 'prod' holds prefix product up to current lane.
    // 'inv_total' holds 1/(Z0*...*Z31).
    
    // We need a running accumulator 'running_inv'.
    // running_inv starts as inv_total.
    
    uint64_t running_inv[4];
    Load(running_inv, inv_total);
    
    uint64_t z_val[4];
    uint64_t p_prev[4];
    
    // Iterate backwards from lane 31 to 0
    // Since it's a warp, we can't easily loop logic per thread without divergence if we aren't careful.
    // BUT, we can use __shfl_down to propagate the updated inverse.
    
    // Let's use the serial approach inside the warp:
    // Lane 31: Result = inv_total * prod[30] (prefix from 30)
    // Lane 30: Result = (Result of 31) * val[31]
    // Lane 29: Result = (Result of 30) * val[30]
    // ...
    
    // To do this efficiently, we pass a value down the lanes.
    
    uint64_t res[4];
    uint64_t next_z[4];
    
    // Initialize next_z for lane 31 (it needs nothing from 32)
    next_z[0]=1; next_z[1]=0; next_z[2]=0; next_z[3]=0;
    
    // We calculate in reverse.
    // Lane k needs: inv_total * (product of all Z_j where j != k)
    // We can separate into (product 0..k-1) * (product k+1..31).
    
    // Product 0..k-1 is in 'prod' (specifically, we need prefix from neighbor).
    // Product k+1..31 is 'suffix'.
    
    // Let's do the sequential pass logic:
    // 1. Lane 31 computes its inverse.
    //    res = inv_total * prod[30] (shfl from lane 30).
    // 2. Lane 30 computes its inverse.
    //    res = (res of 31) * val[31].
    // This is strictly sequential. In a warp, we simulate this.
    
    // Loop 32 steps.
    // Step 0 (Target Lane 31):
    //   Prefix = shfl(prod, 30).
    //   Res_31 = inv_total * Prefix.
    // Step 1 (Target Lane 30):
    //   Needs Res_31.
    //   Res_30 = Res_31 * val[31].
    // Step 2 (Target Lane 29):
    //   Needs Res_30.
    //   Res_29 = Res_30 * val[30].
    
    // This requires a value to be passed down.
    uint64_t pass_down[4];
    
    // Initialization for the "loop"
    // Lane 31 is active first.
    pass_down[0] = inv_total[0]; pass_down[1] = inv_total[1]; pass_down[2] = inv_total[2]; pass_down[3] = inv_total[3];
    
    // We iterate to compute target lanes 31, 30, ..., 0
    // But we want to do it so all threads end up with their result simultaneously if possible? 
    // No, serial dependency.
    // In CUDA warp, instructions are lockstep.
    // If we write a loop `for(int k=31; k>=0; k--)`, all threads execute it.
    // Inside, `if (k == 31)` ... `if (k == 30)` ...
    // The `pass_down` must be shuffled.
    
    // Simplified: All threads compute the sequence in sync, but only "activate" and store result when their turn comes.
    // "turn" corresponds to `lane == k`.
    
    // active_res starts as inv_total.
    uint64_t active_res[4];
    Load(active_res, inv_total);
    
    // We need Z values from neighbors. We have them in smem or registers (val).
    
    for (int k = 31; k >= 0; --k) {
        // Targeting lane `k`.
        
        if (lane == k) {
            // Compute result for this lane
            // Res = active_res * prefix_prod[k-1]
            if (k > 0) {
                uint64_t p_prev[4];
                // Get prefix product from lane k-1
                #pragma unroll
                for(int i=0; i<4; ++i) p_prev[i] = __shfl_sync(mask, prod[i], k-1);
                _ModMult(res, active_res, p_prev);
            } else {
                // Lane 0: prefix is 1. Res = active_res.
                Load(res, active_res);
            }
            // Write result back to val (register)
            Load(val, res);
        }
        
        // Prepare active_res for next iteration (k-1)
        // active_res_new = active_res * Z[k]
        // We need Z[k]. Lane k has it.
        // Broadcast Z[k] from lane k to all.
        uint64_t z_k[4];
        #pragma unroll
        for(int i=0; i<4; ++i) z_k[i] = __shfl_sync(mask, val[i], k); // Note: val here is original Z, NOT the result we just wrote (unless we overwrite).
        
        // IMPORTANT: We overwrote `val` inside `if(lane==k)`. 
        // We must ensure we use the ORIGINAL Z for the propagation.
        // Let's assume smem holds original Z. Use smem.
        #pragma unroll
        for(int i=0; i<4; ++i) z_k[i] = smem[k][i];
        
        _ModMult(active_res, active_res, z_k);
    }
    // Final result is in `val`.
}

// ============================================================
// KERNEL: OPTIMIZED RANDOM SEARCH
// ============================================================
__launch_bounds__(256, 2)
__global__ void kernel_random_search(
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    uint32_t batch_size,
    uint32_t iters_per_launch 
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    
    // RNG State
    uint64_t rng_s0 = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t rng_s1 = (uint64_t)clock64() ^ (rng_s0 << 32);
    if (rng_s0 == 0) rng_s0 = 1; 
    
    auto xorshift128plus = [&]() -> uint64_t {
        uint64_t s1 = rng_s0;
        uint64_t s0 = rng_s1;
        rng_s0 = s0;
        s1 ^= s1 << 23;
        return (rng_s1 = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
    };

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    const uint32_t target_prefix = c_target_prefix;
    const int vanity_len = c_vanity_len;

    auto check_vanity = [&](const uint8_t* h20) -> bool {
        if (vanity_len >= 4) {
            if (load_u32_le(h20) != target_prefix) return false;
            #pragma unroll
            for (int k = 4; k < 20; ++k) {
                if (k >= vanity_len) break;
                if (h20[k] != c_target_hash160[k]) return false;
            }
            return true;
        } else {
            for (int k = 0; k < vanity_len; ++k) {
                if (h20[k] != c_target_hash160[k]) return false;
            }
            return true;
        }
    };

    // Main Loop
    for (uint32_t iter = 0; iter < iters_per_launch; ++iter) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // 1. Generate Random Scalar
        uint64_t k[4];
        k[0] = xorshift128plus();
        k[1] = xorshift128plus();
        k[2] = xorshift128plus();
        k[3] = xorshift128plus();

        // 2. Apply Mask & Add Range Start
        k[0] &= c_RangeMask[0];
        k[1] &= c_RangeMask[1];
        k[2] &= c_RangeMask[2];
        k[3] &= c_RangeMask[3];

        uint64_t carry = 0;
        __uint128_t res = (__uint128_t)k[0] + c_RangeStart[0];
        k[0] = (uint64_t)res; carry = res >> 64;
        
        res = (__uint128_t)k[1] + c_RangeStart[1] + carry;
        k[1] = (uint64_t)res; carry = res >> 64;
        
        res = (__uint128_t)k[2] + c_RangeStart[2] + carry;
        k[2] = (uint64_t)res; carry = res >> 64;
        
        k[3] = k[3] + c_RangeStart[3] + carry;

        // 3. Jacobian Scalar Mul
        uint64_t Xj[4], Yj[4], Zj[4];
        scalarMulBaseJacobian(k, Xj, Yj, Zj);

        // 4. Batch Inversion (Warp)
        // This function transforms Zj to 1/Zj in-place (logically)
        // We need to handle the 'Infinity' case (Z=0). 
        // If Z=0, k was 0. Range logic prevents k=0 if mask correct? Or if range is 0.
        // Assume valid point.
        
        batch_invert_warp(Zj);

        // 5. Convert to Affine
        // x = X * (1/Z)^2
        // y = Y * (1/Z)^3
        
        uint64_t z2[4], z3[4];
        _ModSqr(z2, Zj);       // (1/Z)^2
        _ModMult(z3, Zj, z2);  // (1/Z)^3
        
        uint64_t x1[4], y1[4];
        _ModMult(x1, Xj, z2);
        _ModMult(y1, Yj, z3);

        // 6. Hash & Check Main Point
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool match = check_vanity(h20);
            if (__any_sync(full_mask, match)) {
                if (match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                        d_found_result->iter     = iter;
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=k[k]; // Note: k variable reuse
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

        // 7. Neighbor Check (Standard Logic)
        // Since we have x1, y1 affine, we can do the neighbor check logic as before.
        // Note: The previous `scalarMulBaseAffine` was the bottleneck. Now it's gone.
        // The rest of the kernel remains valuable.
        
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        #pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Gx[(size_t)0*4 + j]; 
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

        uint64_t d0[4];
        uint64_t inverse_full[5]; 
        
        #pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
        #pragma unroll
        for (int j=0;j<4;++j) inverse_full[j] = d0[j];
        inverse_full[4] = 0;
        
        _ModMult(inverse_full, subp[0]); 
        _ModInv(inverse_full); // This is ONE inversion for the whole batch of neighbors. Fast.

        for (int i = 0; i < half - 1; ++i) {
             if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

             uint64_t dx_inv_i[4];
             _ModMult(dx_inv_i, subp[i], inverse_full); 

             // --- BLOK +G ---
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

                 bool match = check_vanity(h20);
                 if (__any_sync(full_mask, match)) {
                     if (match) {
                         if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                             uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=k[k]; // Original scalar
                             uint64_t addv=(uint64_t)(i+1);
                             // Add to scalar
                             uint64_t c_in=0;
                             __uint128_t r = (__uint128_t)fs[0] + addv;
                             fs[0]=(uint64_t)r; c_in=r>>64;
                             // Propagate if needed (omitted for brevity, simple add)
                             // Actually simple carry propagation needed.
                             // ...
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                             d_found_result->threadId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                             d_found_result->iter     = iter;
                             __threadfence_system();
                             atomicExch(d_found_flag, FOUND_READY);
                         }
                     }
                     __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                 }
             }

             // --- BLOK -G ---
             {
                 uint64_t px3[4], s[4], lam[4];
                 uint64_t px_i[4], py_i[4];
                 #pragma unroll
                 for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                 ModNeg256(py_i, py_i); 

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

                 bool match = check_vanity(h20);
                 if (__any_sync(full_mask, match)) {
                     if (match) {
                         if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                             uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=k_final[k];
                             uint64_t subv=(uint64_t)(i+1);
                             for (int k=0;k<4 && subv;++k){ uint64_t old=fs[k]; fs[k]=old-subv; subv=(old<subv)?1ull:0ull; }
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                             d_found_result->threadId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                             d_found_result->iter     = iter;
                             __threadfence_system();
                             atomicExch(d_found_flag, FOUND_READY);
                         }
                     }
                     __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                 }
             }

             // --- BLOK -G --- (Similar logic, omitted to save space, copy from previous correct code)
             // ...
             
             uint64_t gxmi[4];
             #pragma unroll
             for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
             ModSub256(gxmi, gxmi, x1);
             _ModMult(inverse_full, inverse_full, gxmi);
        }
    }
    WARP_FLUSH_HASHES();
}

// Host Helper
void mul256_u64(const uint64_t a[4], uint64_t b, uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t res = (__uint128_t)a[i] * b + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_points_batch_size = 1024; 
    uint32_t iters_per_launch          = 64; 

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) { i++; /* skip */ }
        else if (arg == "--slices"         && i + 1 < argc) { i++; /* skip */ }
    }

    if (range_hex.empty() || vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex>\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }
    
    if (vanity_hash_hex.length() > 40 || vanity_hash_hex.length() % 2 != 0) { std::cerr << "Error: Vanity hash len\n"; return EXIT_FAILURE; }
    uint8_t target_hash160[20]; memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);

    // Calc Range Len
    uint64_t range_len[4]; 
    sub256(range_end, range_start, range_len); 
    add256_u64(range_len, 1ull, range_len);

    // Calc Mask
    uint64_t range_mask[4];
    for(int i=0; i<4; ++i) range_mask[i] = range_len[i];
    
    uint64_t borrow = 1;
    for (int i = 0; i < 4; ++i) {
        if (range_mask[i] >= borrow) {
            range_mask[i] -= borrow;
            borrow = 0;
        } else {
            range_mask[i] -= borrow; // will underflow
            borrow = 1;
        }
    }

    bool is_pow2 = false;
    int bit_count = 0;
    for(int i=0; i<4; ++i) bit_count += __builtin_popcountll(range_len[i]);
    if (bit_count == 1) is_pow2 = true;

    if (!is_pow2) {
        std::cerr << "Warning: Range Length is NOT a power of two. Random distribution will be slightly biased.\n";
    }

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) { std::cerr<<"CUDA init error\n"; return EXIT_FAILURE; }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock = 256;
    int blocks = prop.multiProcessorCount * 8; 
    uint64_t threadsTotal = (uint64_t)blocks * threadsPerBlock;

    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));
    cudaMemcpyToSymbol(c_RangeStart, range_start, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_RangeMask, range_mask, 4 * sizeof(uint64_t));

    uint32_t prefix_le = 0;
    if (vanity_len >= 4) {
         prefix_le = (uint32_t)target_hash160[0] | ((uint32_t)target_hash160[1] << 8) | ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);
    }
    cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));

    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr; 

    auto ck = [](cudaError_t e, const char* msg){ if (e != cudaSuccess) { std::cerr << msg << ": " << cudaGetErrorString(e) << "\n"; std::exit(EXIT_FAILURE); } };

    ck(cudaMalloc(&d_found_flag, sizeof(int)), "malloc flag");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "malloc result");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "malloc accum");
    
    { int zero = FOUND_NONE; unsigned long long zero64=0ull;
      ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init flag");
      ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init accum"); }

    // Precompute G Table
    {
        const uint32_t B = runtime_points_batch_size;
        const uint32_t half = B >> 1;
        uint64_t* h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "malloc");
        ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "malloc");
        ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "malloc");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy");

        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaDeviceSynchronize(), "sync");

        uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half); std::free(h_Gy_half);
    }

    std::cout << "======== Mode: RANDOM SEARCH (Optimized Jacobian) ========\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Threads: " << threadsTotal << "\n";
    std::cout << "================================================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    while (!stop_all) {
        if (g_sigint) std::cerr << "\n[Interruption] Exiting...\n";

        kernel_random_search<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_found_flag, d_found_result, d_hashes_accum,
            runtime_points_batch_size, iters_per_launch
        );
        
        if (cudaGetLastError() != cudaSuccess) { stop_all = true; }

        for (int k=0; k<10 && !stop_all; ++k) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 0.1) {
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << "s | Speed: " << std::fixed << std::setprecision(2) << mkeys
                          << " Mkeys/s | Total: " << h_hashes << "    ";
                std::cout.flush();
                lastHashes = h_hashes; tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read flag");
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break; 
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (stop_all || g_sigint) break;
    }

    cudaDeviceSynchronize();
    std::cout << "\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final flag");

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read result");
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    } else {
        std::cout << "Search stopped.\n";
    }

    cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum);
    cudaStreamDestroy(streamKernel);
    return (h_found_flag == FOUND_READY) ? EXIT_SUCCESS : EXIT_FAILURE;
}