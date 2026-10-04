// Isolated data-feed experiment. Timed kernels keep all 32 milestones and BLAKE3.
#include "cp_cutlass_gemm_types.h"
#include "cp_noise.h"
#include "blake3.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CP_FEED_VARIANT
#define CP_FEED_VARIANT "baseline"
#endif
#ifndef CP_FEED_VERIFY
#define CP_FEED_VERIFY 0
#endif

#define CUDA(call)                                                                                 \
    do                                                                                             \
    {                                                                                              \
        auto e = (call);                                                                           \
        if (e != cudaSuccess)                                                                      \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));           \
    } while (0)
static void status(cutlass::Status s)
{
    if (s != cutlass::Status::kSuccess)
        throw std::runtime_error(cutlassGetStatusString(s));
}
template <class T> struct Buffer
{
    T *p = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n)
    {
        if (n)
            CUDA(cudaMalloc(&p, n * sizeof(T)));
    }
    ~Buffer()
    {
        if (p)
            cudaFree(p);
    }
    Buffer(const Buffer &) = delete;
    std::vector<T> host() const
    {
        std::vector<T> values(count);
        CUDA(cudaMemcpy(values.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};
struct Inputs
{
    int m, n;
    Buffer<int8_t> a, b;
    std::vector<int8_t> ha, hb;
    Inputs(int rows, int columns, int pattern)
        : m(rows), n(columns), a(size_t(m) * 4096), b(size_t(n) * 4096), ha(a.count), hb(b.count)
    {
        uint32_t random = 0x1234567;
        for (auto *v : {&ha, &hb})
            for (auto &x : *v)
            {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                x = pattern == 0 ? 0 : pattern == 1 ? -128 : int8_t(int(random & 255) - 128);
            }
        if (pattern == 3)
        {
            uint8_t sa[32], sb[32], ss[32];
            for (int i = 0; i < 32; ++i)
            {
                sa[i] = i * 7 + 3;
                sb[i] = i * 13 + 9;
                ss[i] = i * 11 + 17;
            }
            std::vector<int8_t> signal(ha.size());
            if (pearl_generate_random_a(ss, 32, m, 4096, signal.data()) ||
                pearl_build_noisy_a(m, 4096, 128, sa, signal.data(), ha.data()) ||
                pearl_build_noisy_b(n, 4096, 128, sb, nullptr, hb.data()))
                throw std::runtime_error("noise fixture failed");
        }
        CUDA(cudaMemcpy(a.p, ha.data(), ha.size(), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(b.p, hb.data(), hb.size(), cudaMemcpyHostToDevice));
    }
};
struct Reference
{
    std::vector<uint32_t> xors, digests;
};
static Reference reference(const Inputs &in, bool sampled)
{
    const size_t tiles = size_t(in.m) * in.n / 64;
    const size_t count = sampled ? 3 : tiles;
    Reference result;
    result.xors.resize(32 * count);
    result.digests.resize(8 * count);
    const int rows[8] = {0, 1, 2, 3, 16, 17, 18, 19};
    const int cols[8] = {0, 1, 2, 3, 32, 33, 34, 35};
    // INT64 reference, independently derived from root inputs. Each output cell
    // belongs to exactly one scattered tile, so exhaustive mode covers all C cells.
    for (size_t sample = 0; sample < count; ++sample)
    {
        size_t tile = sampled ? (sample == 0   ? 0
                                 : sample == 1 ? tiles / 2 + 113
                                               : tiles - 1)
                              : sample;
        size_t cta = tile / 256;
        int row, col;
        MmaLaneTile128x128::thread_cell_global(int(cta / (in.n / 128)) * 128,
                                               int(cta % (in.n / 128)) * 128, int(tile % 256), row,
                                               col);
        int64_t accum[8][8] = {};
        uint32_t words[16] = {};
        for (int step = 0; step < 32; ++step)
        {
            uint32_t x = 0;
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                {
                    for (int k = step * 128; k < (step + 1) * 128; ++k)
                        accum[r][c] += int64_t(in.ha[size_t(row + rows[r]) * 4096 + k]) *
                                       in.hb[size_t(col + cols[c]) * 4096 + k];
                    x ^= uint32_t(accum[r][c]);
                }
            result.xors[size_t(step) * count + sample] = x;
            uint32_t w = words[step % 16];
            words[step % 16] = (w << 13 | w >> 19) ^ x;
        }
        uint8_t key[32];
        std::fill(key, key + 32, 37);
        blake3_hasher hash;
        blake3_hasher_init_keyed(&hash, key);
        blake3_hasher_update(&hash, words, 64);
        blake3_hasher_finalize(&hash,
                               reinterpret_cast<uint8_t *>(result.digests.data() + sample * 8), 32);
    }
    return result;
}
static uint64_t checked_values = 0;
static void compare(const std::vector<uint32_t> &actual, const std::vector<uint32_t> &expected,
                    const char *name)
{
    if (actual.size() != expected.size())
        throw std::runtime_error(std::string(name) + " size mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i])
            throw std::runtime_error(std::string(name) + " mismatch at " + std::to_string(i));
    checked_values += actual.size();
}
template <class T> static void run(Inputs &in, int pattern, int repeats, const char *shape)
{
    const size_t tiles = size_t(in.m) * in.n / 64;
    Buffer<uint32_t> key(8), dump(tiles * 32), digests(CP_FEED_VERIFY ? tiles * 8 : 0);
    Buffer<int> found(1), rows(1), cols(1);
    CUDA(cudaMemset(key.p, 37, 32));
    CUDA(cudaMemset(found.p, 0, 4));
#if CP_FEED_VERIFY
    CUDA(cudaMemcpyToSymbol(cp_feed_digest_sink, &digests.p, sizeof(digests.p)));
    const int tile_cols = in.n / 128;
    CUDA(cudaMemcpyToSymbol(cp_feed_digest_columns, &tile_cols, sizeof(tile_cols)));
    CUDA(cudaMemset(digests.p, 0xa5, digests.count * 4));
#endif
    CpCutlassJackpotLaunch jackpot{};
    jackpot.d_a_key8 = key.p;
    jackpot.d_found = found.p;
    jackpot.d_out_t_rows = rows.p;
    jackpot.d_out_t_cols = cols.p;
    cp_cutlass::FusedMilestoneGemmOp<T> op;
    // Verification and pre-timing sampled checks dump every milestone. Timed
    // invocations reinitialize with a null dump pointer, preserving the full hash.
    status(op.initialize(in.m, in.n, 4096, in.m, in.n, in.a.p, in.b.p, dump.p, in.n / 128, tiles,
                         &jackpot));
    status(op());
    CUDA(cudaDeviceSynchronize());
    auto expected = reference(in, !CP_FEED_VERIFY);
    if (CP_FEED_VERIFY)
    {
        compare(dump.host(), expected.xors, "all milestone XORs");
        compare(digests.host(), expected.digests, "all keyed BLAKE3 digests");
        std::printf("{\"type\":\"correctness\",\"variant\":\"%s\",\"tile\":\"%s\",\"pattern\":%d,"
                    "\"passed\":true}\n",
                    CP_FEED_VARIANT, shape, pattern);
    }
    else
    {
        std::vector<uint32_t> sampled(96);
        for (int step = 0; step < 32; ++step)
            for (int sample = 0; sample < 3; ++sample)
            {
                size_t tile = sample == 0 ? 0 : sample == 1 ? tiles / 2 + 113 : tiles - 1;
                CUDA(cudaMemcpy(&sampled[step * 3 + sample], dump.p + size_t(step) * tiles + tile,
                                4, cudaMemcpyDeviceToHost));
            }
        compare(sampled, expected.xors, "large-panel milestone samples");
        status(op.initialize(in.m, in.n, 4096, in.m, in.n, in.a.p, in.b.p, nullptr, in.n / 128,
                             tiles, &jackpot));
        for (int i = 0; i < 40; ++i)
            status(op());
        CUDA(cudaDeviceSynchronize());
        cudaEvent_t start, end;
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&end));
        std::vector<float> ms;
        std::vector<double> wall;
        for (int i = 0; i < repeats; ++i)
        {
            auto before = std::chrono::steady_clock::now();
            CUDA(cudaEventRecord(start));
            status(op());
            CUDA(cudaEventRecord(end));
            CUDA(cudaEventSynchronize(end));
            float elapsed;
            CUDA(cudaEventElapsedTime(&elapsed, start, end));
            ms.push_back(elapsed);
            wall.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before)
                    .count());
        }
        cudaFuncAttributes attr;
        CUDA(cudaFuncGetAttributes(&attr, cp_cutlass::FusedKernelEntry<T>));
        int blocks;
        const int shared = sizeof(typename T::GemmKernel::SharedStorage);
        CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, cp_cutlass::FusedKernelEntry<T>,
                                                           T::GemmKernel::kThreadCount, shared));
        std::printf("{\"type\":\"measurement\",\"variant\":\"%s\",\"tile\":\"%s\",\"m\":%d,\"n\":%"
                    "d,\"k\":4096,\"registers\":%d,\"local_bytes\":%zu,\"shared_bytes\":%d,"
                    "\"active_blocks_per_sm\":%d,\"gpu_ms\":[",
                    CP_FEED_VARIANT, shape, in.m, in.n, attr.numRegs, attr.localSizeBytes, shared,
                    blocks);
        for (size_t i = 0; i < ms.size(); ++i)
            std::printf("%s%.6f", i ? "," : "", ms[i]);
        std::printf("],\"wall_ms\":[");
        for (size_t i = 0; i < wall.size(); ++i)
            std::printf("%s%.6f", i ? "," : "", wall[i]);
        std::puts("]}");
        CUDA(cudaEventDestroy(start));
        CUDA(cudaEventDestroy(end));
    }
    if (found.host()[0])
        throw std::runtime_error("unexpected zero-target hit");
}
int main(int argc, char **argv)
{
    try
    {
        cudaDeviceProp gpu;
        CUDA(cudaGetDeviceProperties(&gpu, 0));
        if (gpu.major * 10 + gpu.minor != 75)
            throw std::runtime_error("this experiment targets sm_75 only");
        std::printf("{\"type\":\"metadata\",\"gpu\":\"%s\",\"variant\":\"%s\",\"verify\":%s}\n",
                    gpu.name, CP_FEED_VARIANT, CP_FEED_VERIFY ? "true" : "false");
        if (CP_FEED_VERIFY)
        {
            for (int pattern = 0; pattern < 4; ++pattern)
            {
                Inputs in(256, 256, pattern);
                run<cp_cutlass::Gemm128x128TensorOp>(in, pattern, 0, "128x128");
                run<cp_cutlass::Gemm256x128TensorOp>(in, pattern, 0, "256x128");
            }
        }
        else
        {
            int repeats = argc > 1 ? std::stoi(argv[1]) : 60;
            if (repeats < 3 || repeats > 1000 || argc > 3)
                throw std::runtime_error("invalid repeats/arguments");
            Inputs in(4096, 131072, 3);
            if (argc > 2)
                run<cp_cutlass::Gemm256x128TensorOp>(in, 3, repeats, "256x128");
            else
                run<cp_cutlass::Gemm128x128TensorOp>(in, 3, repeats, "128x128");
        }
        std::printf("{\"type\":\"complete\",\"passed\":true,\"checked_values\":%llu}\n",
                    (unsigned long long)checked_values);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
