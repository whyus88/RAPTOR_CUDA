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

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint8_t c_target_hash160[20];
__constant__ int c_vanity_len;
__constant__ uint32_t c_target_prefix;

// Deklarasi Constant untuk Endomorphism dan 8-byte check
__constant__ uint64_t c_beta[4] = {
    0x7ae96a2b657c0710ULL, 0x6e64479eac3434e9ULL,
    0x9cf0497512f58995ULL, 0xc1396c28719501eeULL
};
__constant__ uint32_t c_target_prefix_u32_2;

// =============================================================================
// SHOTGUN + ENDOMORPHISM KERNEL
// =============================================================================

__global__ void kernel_shotgun_hash(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    const uint64_t* __restrict__ scalars,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ d_hashes_accum,
    uint32_t num_samples
)
{
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_samples) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    uint64_t x1[4], y1[4], k[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        x1[i] = Px[gid*4+i];
        y1[i] = Py[gid*4+i];
        k[i]  = scalars[gid*4+i];
    }

    // Hitung Endomorphism: Ex = Beta * x1 mod p
    uint64_t ex[4];
    _ModMult(ex, (uint64_t*)x1, (uint64_t*)c_beta);

    // Fungsi atomic check & store
    auto store_result = [&](const uint64_t* x, const uint64_t* y, int type) {
        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
            #pragma unroll
            for (int i=0;i<4;++i) d_found_result->scalar[i] = k[i];
            #pragma unroll
            for (int i=0;i<4;++i) d_found_result->Rx[i] = x[i];
            
            if (type & 1) { // Jika negasi (-P atau -EndoP)
                uint64_t ny[4]; ModNeg256(ny, (uint64_t*)y);
                #pragma unroll
                for (int i=0;i<4;++i) d_found_result->Ry[i] = ny[i];
            } else {
                #pragma unroll
                for (int i=0;i<4;++i) d_found_result->Ry[i] = y[i];
            }
            d_found_result->threadId = (int)gid;
            d_found_result->iter     = type; // 0:P, 1:-P, 2:EndoP, 3:-EndoP
            __threadfence_system();
            atomicExch(d_found_flag, FOUND_READY);
        }
    };

    // Fungsi hashing & pengecekan cepat
    auto check_hash = [&](const uint64_t* x, const uint64_t* y, bool flip_prefix, int type) -> bool {
        uint8_t odd = (uint8_t)(y[0] & 1ULL);
        if (flip_prefix) odd = !odd;
        uint8_t prefix = odd ? 0x03 : 0x02;
        
        uint8_t h20[20];
        getHash160_33_from_limbs(prefix, x, h20);

        bool match = false;
        if (c_vanity_len >= 8) {
            if (*(const uint32_t*)h20 == c_target_prefix && *(const uint32_t*)(h20 + 4) == c_target_prefix_u32_2) {
                match = true;
                for (int i = 8; i < c_vanity_len; ++i) if (h20[i] != c_target_hash160[i]) { match = false; break; }
            }
        } else if (c_vanity_len >= 4) {
            if (*(const uint32_t*)h20 == c_target_prefix) {
                match = true;
                for (int i = 4; i < c_vanity_len; ++i) if (h20[i] != c_target_hash160[i]) { match = false; break; }
            }
        } else {
            match = true;
            for (int i = 0; i < c_vanity_len; ++i) if (h20[i] != c_target_hash160[i]) { match = false; break; }
        }
        
        if (match) store_result(x, y, type);
        return match;
    };

    // Tambah 4 hash ke akumulasi
    atomicAdd(d_hashes_accum, 4ULL);

    // 1. Cek P (x1, y1)
    if (check_hash(x1, y1, false, 0)) return;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    // 2. Cek -P (x1, -y1) -> prefix di-flip
    if (check_hash(x1, y1, true, 1)) return;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    // 3. Cek EndoP (ex, y1)
    if (check_hash(ex, y1, false, 2)) return;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    // 4. Cek -EndoP (ex, -y1)
    if (check_hash(ex, y1, true, 3)) return;
}

// =============================================================================
// CPU HELPER: Hitung Scalar Exak dari Endomorphism
// =============================================================================

void mul_lambda_mod_n_cpu(const uint64_t k[4], uint64_t out[4]) {
    const uint64_t lambda[4] = {
        0x5363ad4cc05c30e0ULL, 0xa5261c028812645aULL,
        0x122e22ea20816678ULL, 0xdf02967c1b23bd72ULL
    };
    const uint64_t n[4] = {
        0xD0364141ULL, 0xBFD25E8CULL, 0xAF48A03BULL, 0xFFFFFFFFFFFFFFFFULL
    };
    
    __uint128_t t[5] = {0};
    // Hanya loop i=0 dan i=1 karena untuk 72-bit, k[2] dan k[3] pasti 0
    for(int i=0; i<2; ++i) {
        for(int j=0; j<4; ++j) {
            __uint128_t prod = (__uint128_t)k[i] * lambda[j];
            __uint128_t old = t[i+j];
            t[i+j] += prod;
            if(t[i+j] < old) {
                for(int l=i+j+1; l<5; ++l) {
                    if(++t[l] != 0) break;
                }
            }
        }
    }
    
    // Modulus N (karena hasil hanya sampai 320-bit, ini sangat cepat)
    __uint128_t tn[5] = {0};
    uint64_t top = (uint64_t)t[4];
    for(int j=0; j<4; ++j) {
        tn[j] += (__uint128_t)top * n[j];
    }
    for(int i=0; i<4; ++i) {
        if(tn[i] >> 64) {
            tn[i+1] += (tn[i] >> 64);
            tn[i] &= 0xFFFFFFFFFFFFFFFFULL;
        }
    }
    
    uint64_t borrow = 0;
    for(int i=0; i<4; ++i) {
        uint64_t ti = (uint64_t)t[i];
        uint64_t tni = (uint64_t)tn[i];
        uint64_t diff = ti - tni - borrow;
        borrow = (ti < tni + borrow) ? 1 : 0;
        out[i] = diff;
    }
    
    if(borrow) {
        uint64_t c = 0;
        for(int i=0; i<4; ++i) {
            __uint128_t sum = (__uint128_t)out[i] + n[i] + c;
            out[i] = (uint64_t)sum;
            c = (sum >> 64) ? 1 : 0;
        }
    }
}

void get_final_scalar_cpu(const uint64_t k[4], int type, uint64_t out[4]) {
    const uint64_t n[4] = { 0xD0364141ULL, 0xBFD25E8CULL, 0xAF48A03BULL, 0xFFFFFFFFFFFFFFFFULL };
    if (type == 0) {
        for(int i=0;i<4;++i) out[i] = k[i];
    } else if (type == 1) {
        uint64_t borrow = 0;
        for(int i=0; i<4; ++i) {
            uint64_t diff = n[i] - k[i] - borrow;
            borrow = (n[i] < k[i] + borrow) ? 1 : 0;
            out[i] = diff;
        }
    } else if (type == 2) {
        mul_lambda_mod_n_cpu(k, out);
    } else if (type == 3) {
        uint64_t tmp[4];
        mul_lambda_mod_n_cpu(k, tmp);
        uint64_t borrow = 0;
        for(int i=0; i<4; ++i) {
            uint64_t diff = n[i] - tmp[i] - borrow;
            borrow = (n[i] < tmp[i] + borrow) ? 1 : 0;
            out[i] = diff;
        }
    }
}

// =============================================================================
// DEKLARASI EKSTERNAL
// =============================================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// =============================================================================
// MAIN HOST
// =============================================================================

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_batch_size = 65536; // Default 64K samples per launch

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--batch" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<17)) {
                std::cerr << "Error: --batch must be in 1..131072\n"; return EXIT_FAILURE;
            }
            runtime_batch_size = (uint32_t)v;
        }
    }

    if (range_hex.empty() || vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex> [--batch N]\n";
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

    if (vanity_hash_hex.length() > 40 || vanity_hash_hex.length() % 2 != 0) {
        std::cerr << "Error: Vanity hash160 hex length must be even and <= 40 characters.\n";
        return EXIT_FAILURE;
    }
    uint8_t target_hash160[20];
    memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);

    // Hitung Range Length
    uint64_t range_len[4]; 
    sub256(range_end, range_start, range_len); 
    add256_u64(range_len, 1ull, range_len);

    // Setup GPU
    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
    int blocks = (runtime_batch_size + threadsPerBlock - 1) / threadsPerBlock;

    // Hitung len_minus1 untuk masking random 72-bit
    uint64_t len_minus1[4];
    {   uint64_t borrow=1ull;
        for (int i=0;i<4;++i) {
            uint64_t v=range_len[i]-borrow; borrow=(range_len[i]<borrow)?1ull:0ull; len_minus1[i]=v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) len_minus1[k]=range_len[k]; break; }
        }
    }

    // Copy Constants to Device
    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));

    uint32_t prefix_le = 0;
    if (vanity_len >= 4) {
         prefix_le = (uint32_t)target_hash160[0]
                   | ((uint32_t)target_hash160[1] << 8)
                   | ((uint32_t)target_hash160[2] << 16)
                   | ((uint32_t)target_hash160[3] << 24);
    }
    cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));

    uint32_t prefix_le_2 = 0;
    if (vanity_len >= 8) {
         prefix_le_2 = (uint32_t)target_hash160[4]
                     | ((uint32_t)target_hash160[5] << 8)
                     | ((uint32_t)target_hash160[6] << 16)
                     | ((uint32_t)target_hash160[7] << 24);
    }
    cudaMemcpyToSymbol(c_target_prefix_u32_2, &prefix_le_2, sizeof(prefix_le_2));

    // Alokasi Memori (Sangat Kecil dibanding Grasshopper)
    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) { std::cerr << msg << ": " << cudaGetErrorString(e) << "\n"; std::exit(EXIT_FAILURE); }
    };

    uint64_t *d_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr;
    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr;

    size_t batch_bytes = (size_t)runtime_batch_size * 4 * sizeof(uint64_t);
    ck(cudaMalloc(&d_scalars, batch_bytes), "cudaMalloc(d_scalars)");
    ck(cudaMalloc(&d_Px, batch_bytes), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py, batch_bytes), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_found_flag, sizeof(int)), "cudaMalloc(d_found_flag)");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "cudaMalloc(d_found_result)");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "cudaMalloc(d_hashes_accum)");

    uint64_t* h_random_scalars = nullptr;
    cudaHostAlloc(&h_random_scalars, batch_bytes, cudaHostAllocWriteCombined);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

    cudaStream_t stream1, stream2;
    ck(cudaStreamCreate(&stream1), "stream1");
    ck(cudaStreamCreate(&stream2), "stream2");

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Mode"              << " : SHOTGUN 4X ENDOMORPHISM\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "Batch Size"        << " : " << runtime_batch_size << " (" << (runtime_batch_size*4) << " effective keys)\n";
    std::cout << std::left << std::setw(20) << "Vanity Target"     << " : " << vanity_hash_hex << " (" << vanity_len << " bytes)\n\n";
    std::cout << "======== Phase-1: Extreme Random Search =============\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;
    bool stop_all = false;
    uint64_t total_launched = 0;

    while (!stop_all) {
        if (g_sigint) { std::cerr << "\n[Ctrl+C] Interrupt received.\n"; stop_all = true; break; }

        // 1. Generate Random Scalars (Fast CPU task)
        for (uint64_t i = 0; i < runtime_batch_size; ++i) {
            uint64_t rand_offset[4];
            // Optimasi khusus 72-bit: hanya generate limb bawah
            rand_offset[0] = dist(gen);
            rand_offset[1] = dist(gen) & len_minus1[1]; 
            rand_offset[2] = 0; // Karena 72-bit, limb 2 dan 3 pasti 0
            rand_offset[3] = 0;
            
            add256(range_start, rand_offset, &h_random_scalars[i*4]);
        }

        // 2. Reset & Copy
        int zero = FOUND_NONE; 
        unsigned long long zero64 = 0ull;
        ck(cudaMemcpyAsync(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice, stream1), "reset flag");
        ck(cudaMemcpyAsync(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice, stream1), "reset hashes");
        ck(cudaMemcpyAsync(d_scalars, h_random_scalars, batch_bytes, cudaMemcpyHostToDevice, stream1), "cpy scalars");

        // 3. Launch Scalar Multiplication (Gunakan stream yang berbeda agar overlap dengan hash check nanti jika diperlukan,
        //    tapi karena hash bergantung pada Px/Py, kita harus sync disini)
        scalarMulKernelBase<<<blocks, threadsPerBlock, 0, stream1>>>(d_scalars, d_Px, d_Py, runtime_batch_size);
        ck(cudaStreamSynchronize(stream1), "scalarMul sync");
        ck(cudaGetLastError(), "scalarMul launch");

        // 4. Launch Shotgun Hash Check
        kernel_shotgun_hash<<<blocks, threadsPerBlock, 0, stream2>>>(d_Px, d_Py, d_scalars, d_found_flag, d_found_result, d_hashes_accum, runtime_batch_size);
        
        // 5. Monitor Progress (Non-blocking)
        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 0.5) { // Update 2x per detik agar responsif
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpyAsync(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream2), "read hashes");
                cudaStreamSynchronize(stream2); // Pastikan copy sebelum print
                
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << "s | Speed: " << std::setprecision(2) << mkeys
                          << " Mkeys/s | Total: " << h_hashes;
                std::cout.flush();
                lastHashes = h_hashes; tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpyAsync(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost, stream2), "read flag");
            cudaStreamSynchronize(stream2);
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(stream2);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { ck(cudaGetLastError(), "kernel query"); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cudaStreamSynchronize(stream2);
        total_launched += runtime_batch_size;
    }

    cudaDeviceSynchronize();
    std::cout << "\n\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final read");

    int exit_code = EXIT_SUCCESS;

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read result");
        
        // Hitung private key exak berdasarkan tipe endomorphism
        uint64_t final_scalar[4];
        get_final_scalar_cpu(host_result.scalar, host_result.iter, final_scalar);

        std::cout << "======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(final_scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
        std::cout << "Match Type    : " << host_result.iter << " (0:P, 1:-P, 2:EndoP, 3:-EndoP)\n";
    } else {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            exit_code = 130;
        } else {
            std::cout << "======== SEARCH STOPPED ================================\n";
        }
    }

    // Cleanup
    cudaFree(d_scalars); cudaFree(d_Px); cudaFree(d_Py);
    cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum);
    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
    if (h_random_scalars) cudaFreeHost(h_random_scalars);

    return exit_code;
}