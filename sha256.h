#ifndef SHA256_H
#define SHA256_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <string>

// ============================================================
// 1. HOST IMPLEMENTATION (CPU) - Dari kode yang Anda berikan
// ============================================================
namespace host_sha256 {
    static constexpr uint32_t K[64] = {
        0x428A2F98u,0x71374491u,0xB5C0FBCFu,0xE9B5DBA5u,0x3956C25Bu,0x59F111F1u,0x923F82A4u,0xAB1C5ED5u,
        0xD807AA98u,0x12835B01u,0x243185BEu,0x550C7DC3u,0x72BE5D74u,0x80DEB1FEu,0x9BDC06A7u,0xC19BF174u,
        0xE49B69C1u,0xEFBE4786u,0x0FC19DC6u,0x240CA1CCu,0x2DE92C6Fu,0x4A7484AAu,0x5CB0A9DCu,0x76F988DAu,
        0x983E5152u,0xA831C66Du,0xB00327C8u,0xBF597FC7u,0xC6E00BF3u,0xD5A79147u,0x06CA6351u,0x14292967u,
        0x27B70A85u,0x2E1B2138u,0x4D2C6DFCu,0x53380D13u,0x650A7354u,0x766A0ABBu,0x81C2C92Eu,0x92722C85u,
        0xA2BFE8A1u,0xA81A664Bu,0xC24B8B70u,0xC76C51A3u,0xD192E819u,0xD6990624u,0xF40E3585u,0x106AA070u,
        0x19A4C116u,0x1E376C08u,0x2748774Cu,0x34B0BCB5u,0x391C0CB3u,0x4ED8AA4Au,0x5B9CCA4Fu,0x682E6FF3u,
        0x748F82EEu,0x78A5636Fu,0x84C87814u,0x8CC70208u,0x90BEFFFAu,0xA4506CEBu,0xBEF9A3F7u,0xC67178F2u
    };
    
    static inline uint32_t rotr(uint32_t x, uint32_t n){ return (x>>n) | (x<<(32u-n)); }
    static inline uint32_t Ch  (uint32_t x,uint32_t y,uint32_t z){ return (x & y) ^ (~x & z); }
    static inline uint32_t Maj (uint32_t x,uint32_t y,uint32_t z){ return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint32_t BSIG0(uint32_t x){ return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
    static inline uint32_t BSIG1(uint32_t x){ return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
    static inline uint32_t SSIG0(uint32_t x){ return rotr(x,7) ^ rotr(x,18) ^ (x>>3); }
    static inline uint32_t SSIG1(uint32_t x){ return rotr(x,17)^ rotr(x,19) ^ (x>>10); }

    static void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
        uint32_t H[8] = {
            0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
            0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
        };

        uint8_t block[64] = {0};
        size_t n = len;

        auto process = [&](const uint8_t b[64]){
            uint32_t W[64];
            for (int i=0;i<16;++i){
                W[i] = ( (uint32_t)b[4*i+0]<<24 )|
                       ( (uint32_t)b[4*i+1]<<16 )|
                       ( (uint32_t)b[4*i+2]<<8  )|
                       ( (uint32_t)b[4*i+3]<<0  );
            }
            for (int t=16;t<64;++t)
                W[t] = SSIG1(W[t-2]) + W[t-7] + SSIG0(W[t-15]) + W[t-16];

            uint32_t a=H[0],b_=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
            for (int t=0;t<64;++t){
                uint32_t T1 = h + BSIG1(e) + Ch(e,f,g) + K[t] + W[t];
                uint32_t T2 = BSIG0(a) + Maj(a,b_,c);
                h=g; g=f; f=e; e=d+T1; d=c; c=b_; b_=a; a=T1+T2;
            }
            H[0]+=a; H[1]+=b_; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
        };

        std::memset(block, 0, 64);
        if (n) std::memcpy(block, data, n);
        block[n] = 0x80;
        uint64_t bitlen = (uint64_t)len * 8ull;
        block[63] = (uint8_t)(bitlen      );
        block[62] = (uint8_t)(bitlen >> 8 );
        block[61] = (uint8_t)(bitlen >> 16);
        block[60] = (uint8_t)(bitlen >> 24);
        block[59] = (uint8_t)(bitlen >> 32);
        block[58] = (uint8_t)(bitlen >> 40);
        block[57] = (uint8_t)(bitlen >> 48);
        block[56] = (uint8_t)(bitlen >> 56);
        process(block);

        for (int i=0;i<8;++i){
            out[4*i+0] = (uint8_t)(H[i] >> 24);
            out[4*i+1] = (uint8_t)(H[i] >> 16);
            out[4*i+2] = (uint8_t)(H[i] >> 8 );
            out[4*i+3] = (uint8_t)(H[i] >> 0 );
        }
    }

    static void sha256d(const uint8_t* data, size_t len, uint8_t out[32]){
        uint8_t tmp[32];
        sha256(data, len, tmp);
        sha256(tmp, 32, out);
    }
} 

static bool base58_decode(const std::string& in, std::vector<uint8_t>& out)
{
    static const char* ALPH = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    static int8_t map[128];
    static bool inited=false;
    if (!inited){
        std::fill(std::begin(map), std::end(map), (int8_t)-1);
        for (int i=0;i<58;++i) map[(unsigned char)ALPH[i]] = (int8_t)i;
        inited=true;
    }
    if (in.empty()) return false;
    size_t zeros = 0;
    while (zeros < in.size() && in[zeros] == '1') ++zeros;

    std::vector<uint8_t> b256; b256.reserve(in.size()*733/1000 + 1); 
    for (char ch : in) {
        unsigned char uc = (unsigned char)ch;
        if (uc >= 128 || map[uc] == -1) return false;
        int carry = map[uc];
        for (size_t j=0;j<b256.size();++j) {
            int x = (int)b256[j] * 58 + carry;
            b256[j] = (uint8_t)(x & 0xFF);
            carry = x >> 8;
        }
        while (carry) {
            b256.push_back((uint8_t)(carry & 0xFF));
            carry >>= 8;
        }
    }
    out.clear();
    out.resize(zeros, 0u);
    for (auto it=b256.rbegin(); it!=b256.rend(); ++it) out.push_back(*it);
    return true;
}

static bool decode_p2pkh_address(const std::string& addr, uint8_t out_hash160[20])
{
    if (addr.empty() || addr[0] != '1') return false;

    std::vector<uint8_t> raw;
    if (!base58_decode(addr, raw)) return false;
    if (raw.size() != 25) return false;
    if (raw[0] != 0x00) return false;

    uint8_t check[32];
    host_sha256::sha256d(raw.data(), 21, check);
    if ( !std::equal(check, check+4, raw.data()+21) ) return false;

    std::memcpy(out_hash160, raw.data()+1, 20);
    return true;
}

// ============================================================
// 2. DEVICE IMPLEMENTATION (GPU) - Untuk CUDA Kernel
// ============================================================
#ifdef __CUDACC__

// Konstanta K untuk Device
__device__ __constant__ static const uint32_t d_K[64] = {
    0x428A2F98u,0x71374491u,0xB5C0FBCFu,0xE9B5DBA5u,0x3956C25Bu,0x59F111F1u,0x923F82A4u,0xAB1C5ED5u,
    0xD807AA98u,0x12835B01u,0x243185BEu,0x550C7DC3u,0x72BE5D74u,0x80DEB1FEu,0x9BDC06A7u,0xC19BF174u,
    0xE49B69C1u,0xEFBE4786u,0x0FC19DC6u,0x240CA1CCu,0x2DE92C6Fu,0x4A7484AAu,0x5CB0A9DCu,0x76F988DAu,
    0x983E5152u,0xA831C66Du,0xB00327C8u,0xBF597FC7u,0xC6E00BF3u,0xD5A79147u,0x06CA6351u,0x14292967u,
    0x27B70A85u,0x2E1B2138u,0x4D2C6DFCu,0x53380D13u,0x650A7354u,0x766A0ABBu,0x81C2C92Eu,0x92722C85u,
    0xA2BFE8A1u,0xA81A664Bu,0xC24B8B70u,0xC76C51A3u,0xD192E819u,0xD6990624u,0xF40E3585u,0x106AA070u,
    0x19A4C116u,0x1E376C08u,0x2748774Cu,0x34B0BCB5u,0x391C0CB3u,0x4ED8AA4Au,0x5B9CCA4Fu,0x682E6FF3u,
    0x748F82EEu,0x78A5636Fu,0x84C87814u,0x8CC70208u,0x90BEFFFAu,0xA4506CEBu,0xBEF9A3F7u,0xC67178F2u
};

__device__ __forceinline__ uint32_t d_rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }
__device__ __forceinline__ uint32_t d_Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t d_Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t d_BSIG0(uint32_t x) { return d_rotr(x, 2) ^ d_rotr(x, 13) ^ d_rotr(x, 22); }
__device__ __forceinline__ uint32_t d_BSIG1(uint32_t x) { return d_rotr(x, 6) ^ d_rotr(x, 11) ^ d_rotr(x, 25); }
__device__ __forceinline__ uint32_t d_SSIG0(uint32_t x) { return d_rotr(x, 7) ^ d_rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t d_SSIG1(uint32_t x) { return d_rotr(x, 17) ^ d_rotr(x, 19) ^ (x >> 10); }

// Fungsi SHA256 untuk GPU (Optimized for short inputs like Minikeys)
// Asumsi: len <= 55 bytes (single block processing) - cukup untuk Minikey (22-23 chars)
__device__ void sha256_device(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint32_t H[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    uint8_t block[64];
    // Clear block
    for(int i=0; i<64; ++i) block[i] = 0;
    
    // Copy data
    for(size_t i=0; i<len; ++i) block[i] = data[i];
    
    // Padding
    block[len] = 0x80;
    
    // Length (bits) - Big Endian at the end of block
    uint64_t bitlen = len * 8;
    // block[60..63] stores the 64-bit length.
    // Since we cleared the block, we just set the last bytes.
    block[63] = (uint8_t)(bitlen);
    block[62] = (uint8_t)(bitlen >> 8);
    block[61] = (uint8_t)(bitlen >> 16);
    block[60] = (uint8_t)(bitlen >> 24);
    // For len < 55, high bytes of length are 0, which matches the cleared block.
    // Actually, standard sha256 length is 64-bit, so if we want to be strictly correct:
    // block[56..59] are 0 because len < 55 bytes implies bitlen < 440 bits.
    // So bitlen >> 32 is 0. Since block is zeroed, this is handled.

    // Process Block
    uint32_t W[64];
    
    // Prepare W
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[4 * i + 0] << 24) |
               ((uint32_t)block[4 * i + 1] << 16) |
               ((uint32_t)block[4 * i + 2] << 8)  |
               ((uint32_t)block[4 * i + 3] << 0);
    }
    for (int t = 16; t < 64; ++t) {
        W[t] = d_SSIG1(W[t - 2]) + W[t - 7] + d_SSIG0(W[t - 15]) + W[t - 16];
    }

    // Compress
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + d_BSIG1(e) + d_Ch(e, f, g) + d_K[t] + W[t];
        uint32_t T2 = d_BSIG0(a) + d_Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;

    // Output
    for (int i = 0; i < 8; ++i) {
        out[4 * i + 0] = (uint8_t)(H[i] >> 24);
        out[4 * i + 1] = (uint8_t)(H[i] >> 16);
        out[4 * i + 2] = (uint8_t)(H[i] >> 8);
        out[4 * i + 3] = (uint8_t)(H[i] >> 0);
    }
}

#endif // __CUDACC__

#endif // SHA256_H