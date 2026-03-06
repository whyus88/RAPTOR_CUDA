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
#include "CUDAHash.cuh"
#include "CUDAUtils.h"

// ============================================================
// DEFINISI KONSTANTA
// ============================================================
#ifndef FOUND_NONE
#define FOUND_NONE  0
#endif
#ifndef FOUND_LOCK
#define FOUND_LOCK  1
#endif
#ifndef FOUND_READY
#define FOUND_READY 2
#endif

#define MAX_MINIKEY_LEN 32

struct FoundResult {
    int      threadId;
    int      iter;
    char     minikey_str[MAX_MINIKEY_LEN];
    uint64_t scalar[4];
    uint64_t Rx[4];
    uint64_t Ry[4];
};

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

__constant__ uint8_t  c_target_hash160[20];
__constant__ uint32_t c_target_prefix;
__constant__ int      c_vanity_len;
__constant__ int      c_minikey_len_target;
__constant__ const char c_b58_alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// SHA256 Constants
__constant__ const uint32_t c_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// ============================================================
// DEVICE HELPERS
// ============================================================
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t sha2_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t sha2_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t sha2_ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
__device__ __forceinline__ uint32_t sha2_ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
__device__ __forceinline__ uint32_t sha2_sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sha2_sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

__device__ void sha256_device(const uint8_t* data, size_t len, uint8_t hash[32]) {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint32_t w[64];
    
    for (int i = 0; i < 16; ++i) w[i] = 0;
    for (size_t i = 0; i < len; ++i) w[i / 4] |= ((uint32_t)data[i]) << (24 - (i % 4) * 8);
    
    w[len / 4] |= ((uint32_t)0x80) << (24 - (len % 4) * 8);
    uint64_t bitLen = len * 8;
    w[14] = (uint32_t)(bitLen >> 32);
    w[15] = (uint32_t)bitLen;

    for (int i = 16; i < 64; ++i) w[i] = sha2_sig1(w[i-2]) + w[i-7] + sha2_sig0(w[i-15]) + w[i-16];

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], h_loc = h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h_loc + sha2_ep1(e) + sha2_ch(e, f, g) + c_sha256_k[i] + w[i];
        uint32_t t2 = sha2_ep0(a) + sha2_maj(a, b, c);
        h_loc = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h_loc;

    for (int i = 0; i < 8; ++i) {
        hash[i*4 + 0] = (h[i] >> 24) & 0xFF; hash[i*4 + 1] = (h[i] >> 16) & 0xFF;
        hash[i*4 + 2] = (h[i] >> 8) & 0xFF;  hash[i*4 + 3] = (h[i]) & 0xFF;
    }
}

__device__ uint64_t xorshift128plus(uint64_t* s) {
    uint64_t x = s[0]; uint64_t const y = s[1];
    s[0] = y; x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) { return *((const volatile int*)p); }

// Helper Add Mod P (Simple)
__device__ void add_mod_p(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    UADDO(r[0], a[0], b[0]);
    UADDC(r[1], a[1], b[1]);
    UADDC(r[2], a[2], b[2]);
    UADD(r[3], a[3], b[3]);

    bool needs_sub = false;
    if (r[3] > 0xFFFFFFFEFFFFFFFFULL) needs_sub = true;
    else if (r[3] == 0xFFFFFFFEFFFFFFFFULL) {
         if (r[2] > 0xFFFFFFFFFFFFFFFFULL) needs_sub = true;
         else if (r[2] == 0xFFFFFFFFFFFFFFFFULL) {
             if (r[1] > 0xFFFFFFFFFFFFFFFFULL) needs_sub = true;
             else if (r[1] == 0xFFFFFFFFFFFFFFFFULL) {
                 if (r[0] >= 0xFFFFFFFEFFFFFC2FULL) needs_sub = true;
             }
         }
    }
    if (needs_sub) {
        USUBO(r[0], r[0], 0xFFFFFFFEFFFFFC2FULL);
        USUBC(r[1], r[1], 0xFFFFFFFFFFFFFFFFULL);
        USUBC(r[2], r[2], 0xFFFFFFFFFFFFFFFFULL);
        USUB(r[3], r[3], 0xFFFFFFFFFFFFFFFFULL);
    }
}

// ============================================================
// DEVICE: JACOBIAN ELLIPTIC CURVE MATH
// ============================================================

__device__ void pointDoubleJac(uint64_t X[4], uint64_t Y[4], uint64_t Z[4]) {
    if ((Y[0] | Y[1] | Y[2] | Y[3]) == 0) return;

    uint64_t A[4], B[4], C[4], D[4], E[4], F[4], X3[4], Y3[4], Z3[4];

    _ModSqr(A, X);
    _ModSqr(B, Y);
    _ModSqr(C, B);

    uint64_t t1[4];
    add_mod_p(t1, X, B); 
    _ModSqr(D, t1);
    Sub2(D, D, A);
    Sub2(D, D, C);
    add_mod_p(D, D, D);

    add_mod_p(E, A, A);
    add_mod_p(E, E, A);

    _ModSqr(F, E);

    Sub2(X3, F, D);
    Sub2(X3, X3, D);

    Sub2(Y3, D, X3);
    _ModMult(Y3, E, Y3);
    add_mod_p(t1, C, C);
    add_mod_p(t1, t1, t1);
    add_mod_p(t1, t1, t1);
    Sub2(Y3, Y3, t1);

    _ModMult(Z3, Y, Z);
    add_mod_p(Z3, Z3, Z3);

    Load(X, X3); Load(Y, Y3); Load(Z, Z3);
}

// FIX: Menghapus 'const' dari X2 dan Y2 agar sesuai dengan signature _ModMult di CUDAMath.h
__device__ void pointAddMixedJac(uint64_t X1[4], uint64_t Y1[4], uint64_t Z1[4], 
                                 uint64_t X2[4], uint64_t Y2[4]) {
    if ((Z1[0] | Z1[1] | Z1[2] | Z1[3]) == 0) {
        Load(X1, X2); Load(Y1, Y2);
        Z1[0]=1; Z1[1]=0; Z1[2]=0; Z1[3]=0;
        return;
    }
    
    uint64_t Z1Z1[4], U2[4], S2[4], H[4], I[4], J[4], r[4], V[4], X3[4], Y3[4], Z3[4];
    
    _ModSqr(Z1Z1, Z1);
    _ModMult(U2, X2, Z1Z1); // Fixed: X2 is now non-const
    _ModMult(S2, Y2, Z1);   // Fixed: Y2 is now non-const
    _ModMult(S2, S2, Z1Z1);
    
    Sub2(H, U2, X1);
    add_mod_p(I, H, H);
    _ModSqr(I, I);
    
    _ModMult(J, H, I);
    
    Sub2(r, S2, Y1);
    add_mod_p(r, r, r);
    
    _ModMult(V, X1, I);
    
    _ModSqr(X3, r);
    Sub2(X3, X3, J);
    Sub2(X3, X3, V);
    Sub2(X3, X3, V);
    
    Sub2(Y3, V, X3);
    _ModMult(Y3, r, Y3);
    uint64_t t[4];
    _ModMult(t, Y1, J);
    Sub2(Y3, Y3, t);
    
    add_mod_p(Z3, Z1, H);
    _ModSqr(Z3, Z3);
    Sub2(Z3, Z3, Z1Z1);
    Sub2(Z3, Z3, I);
    
    Load(X1, X3); Load(Y1, Y3); Load(Z1, Z3);
}

__device__ void scalarMulJac(const uint64_t k[4], uint64_t Rx_out[4], uint64_t Ry_out[4]) {
    // FIX: Menghapus 'const' agar bisa passed ke pointAddMixedJac
    uint64_t Gx[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
    uint64_t Gy[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

    uint64_t Rx[4] = {0}, Ry[4] = {0}, Rz[4] = {0};
    
    bool started = false;
    int msb = 255;
    while(msb >= 0) {
        int limb = msb >> 6;
        int bit = msb & 63;
        if (k[limb] & (1ULL << bit)) break;
        msb--;
    }
    if (msb < 0) { 
        Rx_out[0]=0; Rx_out[1]=0; Rx_out[2]=0; Rx_out[3]=0;
        Ry_out[0]=0; Ry_out[1]=0; Ry_out[2]=0; Ry_out[3]=0;
        return;
    }

    for (int i = msb; i >= 0; --i) {
        int limb = i >> 6;
        int bit = i & 63;
        
        if (started) {
            pointDoubleJac(Rx, Ry, Rz);
        }
        
        if (k[limb] & (1ULL << bit)) {
            if (!started) {
                Load(Rx, Gx); Load(Ry, Gy);
                Rz[0] = 1; Rz[1]=0; Rz[2]=0; Rz[3]=0;
                started = true;
            } else {
                pointAddMixedJac(Rx, Ry, Rz, Gx, Gy);
            }
        }
    }
    
    if ((Rz[0] | Rz[1] | Rz[2] | Rz[3]) == 0) {
        Rx_out[0]=0; Ry_out[0]=0;
    } else {
        uint64_t zinv[5], z2[4], z3[4];
        Load(zinv, Rz); zinv[4]=0;
        _ModInv(zinv); 
        
        _ModSqr(z2, zinv);
        _ModMult(z3, z2, zinv);
        
        _ModMult(Rx_out, Rx, z2);
        _ModMult(Ry_out, Ry, z3);
    }
}

// ============================================================
// KERNEL
// ============================================================
__launch_bounds__(256, 2)
__global__ void kernel_minikey_search(
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ attempts_accum,
    unsigned long long seed_high
) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = (unsigned)(threadIdx.x & 31);

    uint64_t rng_state[2];
    rng_state[0] = gid ^ 0xDEADBEEFCAFEBABEULL;
    rng_state[1] = (seed_high << 32) ^ gid; 

    const int target_len = c_minikey_len_target;
    char mk_buffer[MAX_MINIKEY_LEN + 2];
    unsigned int local_attempts = 0;

    while (true) {
        if (load_found_flag_relaxed(d_found_flag) == FOUND_READY) break;
        
        mk_buffer[0] = 'S';
        for (int i = 1; i < target_len; ++i) {
            uint64_t r = xorshift128plus(rng_state);
            mk_buffer[i] = c_b58_alphabet[r % 58];
        }
        mk_buffer[target_len] = '\0';

        uint8_t hash_check[32];
        mk_buffer[target_len] = '?';
        sha256_device((uint8_t*)mk_buffer, target_len + 1, hash_check);

        if (hash_check[0] != 0x00) {
            if (++local_attempts >= 1024) {
                if (lane == 0) atomicAdd(attempts_accum, (unsigned long long)local_attempts);
                local_attempts = 0;
            }
            continue;
        }

        mk_buffer[target_len] = '\0';
        uint8_t priv_key_bytes[32];
        sha256_device((uint8_t*)mk_buffer, target_len, priv_key_bytes);

        uint64_t scalar_le[4];
        scalar_le[0] = ((uint64_t)priv_key_bytes[31] << 0) | ((uint64_t)priv_key_bytes[30] << 8) |
                       ((uint64_t)priv_key_bytes[29] << 16) | ((uint64_t)priv_key_bytes[28] << 24) |
                       ((uint64_t)priv_key_bytes[27] << 32) | ((uint64_t)priv_key_bytes[26] << 40) |
                       ((uint64_t)priv_key_bytes[25] << 48) | ((uint64_t)priv_key_bytes[24] << 56);
        scalar_le[1] = ((uint64_t)priv_key_bytes[23] << 0) | ((uint64_t)priv_key_bytes[22] << 8) |
                       ((uint64_t)priv_key_bytes[21] << 16) | ((uint64_t)priv_key_bytes[20] << 24) |
                       ((uint64_t)priv_key_bytes[19] << 32) | ((uint64_t)priv_key_bytes[18] << 40) |
                       ((uint64_t)priv_key_bytes[17] << 48) | ((uint64_t)priv_key_bytes[16] << 56);
        scalar_le[2] = ((uint64_t)priv_key_bytes[15] << 0) | ((uint64_t)priv_key_bytes[14] << 8) |
                       ((uint64_t)priv_key_bytes[13] << 16) | ((uint64_t)priv_key_bytes[12] << 24) |
                       ((uint64_t)priv_key_bytes[11] << 32) | ((uint64_t)priv_key_bytes[10] << 40) |
                       ((uint64_t)priv_key_bytes[9] << 48) | ((uint64_t)priv_key_bytes[8] << 56);
        scalar_le[3] = ((uint64_t)priv_key_bytes[7] << 0) | ((uint64_t)priv_key_bytes[6] << 8) |
                       ((uint64_t)priv_key_bytes[5] << 16) | ((uint64_t)priv_key_bytes[4] << 24) |
                       ((uint64_t)priv_key_bytes[3] << 32) | ((uint64_t)priv_key_bytes[2] << 40) |
                       ((uint64_t)priv_key_bytes[1] << 48) | ((uint64_t)priv_key_bytes[0] << 56);

        uint64_t Rx[4], Ry[4];
        scalarMulJac(scalar_le, Rx, Ry);

        uint8_t h20[20];
        uint8_t prefix = (uint8_t)(Ry[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, Rx, h20);

        if (lane == 0) atomicAdd(attempts_accum, (unsigned long long)local_attempts + 1);
        local_attempts = 0;

        bool match = false;
        if (load_u32_le(h20) == c_target_prefix) {
            match = true;
            for (int k = 4; k < c_vanity_len; ++k) if (h20[k] != c_target_hash160[k]) { match = false; break; }
        }

        if (match) {
            if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                memcpy(d_found_result->minikey_str, mk_buffer, target_len + 1);
                memcpy(d_found_result->scalar, scalar_le, 32);
                memcpy(d_found_result->Rx, Rx, 32);
                memcpy(d_found_result->Ry, Ry, 32);
                __threadfence_system();
                atomicExch(d_found_flag, FOUND_READY);
            }
        }
    }
}

// ============================================================
// HOST CODE
// ============================================================
extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::string vanity_hash_hex;
    int minikey_len = 30;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--minikey-len" && i + 1 < argc) minikey_len = std::atoi(argv[++i]);
    }

    if (vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0] << " --vanity-hash160 <hash> [--minikey-len 30]\n";
        return EXIT_FAILURE;
    }

    uint8_t target_hash160[20]; memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);

    int device=0; cudaDeviceProp prop{};
    cudaGetDevice(&device); cudaGetDeviceProperties(&prop, device);
    
    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));
    cudaMemcpyToSymbol(c_minikey_len_target, &minikey_len, sizeof(int));

    uint32_t prefix_le = 0;
    if (vanity_len >= 4) prefix_le = (uint32_t)target_hash160[0] | ((uint32_t)target_hash160[1] << 8) | ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);
    cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));

    int *d_found_flag; FoundResult *d_found_result; unsigned long long *d_attempts;
    cudaMalloc(&d_found_flag, sizeof(int)); cudaMalloc(&d_found_result, sizeof(FoundResult)); cudaMalloc(&d_attempts, sizeof(unsigned long long));
    int zero = FOUND_NONE; unsigned long long zero_ull = 0;
    cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice); cudaMemcpy(d_attempts, &zero_ull, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    std::cout << "======== MINIKEY SEARCH (Jacobian Optimized) =========\n";
    std::cout << "Device       : " << prop.name << "\n";
    std::cout << "Target Hash  : " << vanity_hash_hex << "\n";
    std::cout << "======== SEARCH STARTED =========\n";

    int threads = 256; int blocks = prop.multiProcessorCount * 4;
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t seed_high = (uint64_t)now.time_since_epoch().count();

    kernel_minikey_search<<<blocks, threads>>>(d_found_flag, d_found_result, d_attempts, seed_high);
    
    auto t0 = std::chrono::high_resolution_clock::now(); auto tLast = t0; unsigned long long last_attempts = 0;
    while (!g_sigint) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int h_flag = 0; cudaMemcpy(&h_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_flag == FOUND_READY) break;

        auto tNow = std::chrono::high_resolution_clock::now(); double dt = std::chrono::duration<double>(tNow - tLast).count();
        if (dt >= 1.0) {
            unsigned long long curr_att = 0; cudaMemcpy(&curr_att, d_attempts, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            double rate = ((double)(curr_att - last_attempts)) / (dt * 1e6);
            std::cout << "\rSpeed: " << std::fixed << std::setprecision(2) << rate << " MCheck/s | Total: " << curr_att << std::flush;
            last_attempts = curr_att; tLast = tNow;
        }
    }

    cudaDeviceSynchronize(); std::cout << "\n";
    int h_flag = 0; cudaMemcpy(&h_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_flag == FOUND_READY) {
        FoundResult res; cudaMemcpy(&res, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
        std::cout << "======== FOUND MATCH! =================================\n";
        std::cout << "Minikey       : " << res.minikey_str << "\n";
        std::cout << "Private Key   : " << formatHex256(res.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(res.Rx, res.Ry) << "\n";
    } else { std::cout << "Search Interrupted.\n"; }

    cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_attempts);
    return 0;
}