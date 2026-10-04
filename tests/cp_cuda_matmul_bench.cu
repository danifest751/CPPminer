// Exact GPU multiplication prototypes; isolated from the miner and pool.
#include "cp_cutlass_gemm_types.h"
#include "cp_noise.h"
#include "blake3.h"
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
};
struct View
{
    const int16_t *p;
    int rows, k, stride, lo, hi;
    View sub(int i, int j, int m, int n) const
    {
        return {p + size_t(i) * stride + j, m, n, stride, lo, hi};
    }
    bool int8() const
    {
        return lo >= -128 && hi <= 127;
    }
};
struct Out
{
    uint32_t *p;
    int rows, cols, stride;
};
__global__ void extend(const int8_t *in, int16_t *out, size_t n)
{
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < n)
        out[i] = in[i];
}
__global__ void add_operands(View a, View b, int16_t *out, int sign)
{
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < size_t(a.rows) * a.k)
        out[i] = int16_t(int(a.p[(i / a.k) * a.stride + i % a.k]) +
                         sign * int(b.p[(i / a.k) * b.stride + i % b.k]));
}
__global__ void split(View in, int8_t *low, int8_t *high)
{
    // Exact signed radix-256 decomposition: x = low + 256 * high.
    // With INT8 roots and at most three recursion levels, even the larger
    // Strassen-Winograd operands fit INT16 and both limbs fit signed INT8.
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(in.rows) * in.k)
        return;
    int x = in.p[(i / in.k) * in.stride + i % in.k], l = x & 255;
    if (l >= 128)
        l -= 256;
    low[i] = int8_t(l);
    high[i] = int8_t((x - l) / 256);
}
__global__ void recombine(const uint32_t *p0, const uint32_t *p1, const uint32_t *p2,
                          const uint32_t *p3, const uint32_t *p4, const uint32_t *p5,
                          const uint32_t *p6, Out out, bool winograd)
{
    int m = out.rows / 2, n = out.cols / 2;
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(m) * n)
        return;
    uint32_t a = p0[i], b = p1[i], c = p2[i], d = p3[i], e = p4[i], f = p5[i], g = p6[i];
    size_t pos = (i / n) * out.stride + i % n;
    if (!winograd)
    {
        out.p[pos] = a + d - e + g;
        out.p[pos + n] = c + e;
        out.p[pos + size_t(m) * out.stride] = b + d;
        out.p[pos + size_t(m) * out.stride + n] = a - b + c + f;
    }
    else
    {
        uint32_t u2 = a + f, u3 = u2 + g, u4 = u2 + e;
        out.p[pos] = a + b;
        out.p[pos + n] = u4 + c;
        out.p[pos + size_t(m) * out.stride] = u3 - d;
        out.p[pos + size_t(m) * out.stride + n] = u3 + e;
    }
}
__global__ void accumulate(uint32_t *c, const uint32_t *delta, size_t count)
{
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count)
        c[i] += delta[i];
}
__global__ void pair_factors(View a, View b, uint32_t *fa, uint32_t *fb)
{
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= a.rows + b.rows)
        return;
    View v = i < a.rows ? a : b;
    int row = i < a.rows ? i : i - a.rows;
    uint32_t x = 0;
    for (int k = 0; k + 1 < v.k; k += 2)
        x += uint32_t(int(v.p[size_t(row) * v.stride + k]) *
                      int(v.p[size_t(row) * v.stride + k + 1]));
    (i < a.rows ? fa : fb)[row] = x;
}
template <bool Pair>
__global__ void simt(View a, View b, Out c, const uint32_t *fa, const uint32_t *fb)
{
    // Shared operand reuse: each 16x16 block produces 256 output values.
    __shared__ int16_t sa[16][32], sb[16][32];
    int x = threadIdx.x, y = threadIdx.y, row = blockIdx.y * 16 + y, col = blockIdx.x * 16 + x;
    uint32_t sum = 0;
    for (int base = 0; base < a.k; base += 32)
    {
        for (int h = 0; h < 2; ++h)
        {
            int k = base + x + 16 * h;
            sa[y][x + 16 * h] = (row < a.rows && k < a.k) ? a.p[size_t(row) * a.stride + k] : 0;
            sb[y][x + 16 * h] = (blockIdx.x * 16 + y < b.rows && k < a.k)
                                    ? b.p[size_t(blockIdx.x * 16 + y) * b.stride + k]
                                    : 0;
        }
        __syncthreads();
        for (int k = 0; k < 32; k += 2)
        {
            int a0 = sa[y][k], a1 = sa[y][k + 1], b0 = sb[x][k], b1 = sb[x][k + 1];
            if (Pair)
                sum += uint32_t((a0 + b1) * (a1 + b0));
            else
                sum += uint32_t(a0 * b0) + uint32_t(a1 * b1);
        }
        __syncthreads();
    }
    if (row < a.rows && col < b.rows)
        c.p[size_t(row) * c.stride + col] = Pair ? sum - fa[row] - fb[col] : sum;
}
__global__ void fold_tiles(const uint32_t *c, int m, int n, int step, uint32_t *words,
                           uint32_t *dump)
{
    size_t tile = size_t(blockIdx.x) * 256 + threadIdx.x, count = size_t(m) * n / 64;
    if (tile >= count)
        return;
    size_t cta = tile / 256;
    int row, col;
    MmaLaneTile128x128::thread_cell_global(int(cta / (n / 128)) * 128, int(cta % (n / 128)) * 128,
                                           int(tile % 256), row, col);
    const int r[8] = {0, 1, 2, 3, 16, 17, 18, 19}, s[8] = {0, 1, 2, 3, 32, 33, 34, 35};
    uint32_t x = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            x ^= c[size_t(row + r[i]) * n + col + s[j]];
    cp_cutlass_jackpot_fold_step(words + tile * 16, step, x);
    if (dump)
        dump[size_t(step) * count + tile] = x;
}
__global__ void finalize(const uint32_t *words, const uint32_t *key, size_t tiles, int n,
                         int *found, int *rows, int *cols, uint32_t *digests)
{
    size_t tile = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (tile >= tiles)
        return;
    uint32_t bound[8] = {};
    size_t cta = tile / 256;
    cp_cutlass_jackpot_try(words + tile * 16, key, bound, int(cta / (n / 128)),
                           int(cta % (n / 128)), int(tile % 256), found, rows, cols);
    if (digests)
        b3_compress64(key, words + tile * 16, digests + tile * 8);
}
__global__ void capture_samples(const uint32_t *c, int n, size_t tiles, int step, uint32_t *out)
{
    int sample = threadIdx.x;
    if (sample >= 3)
        return;
    size_t tile = sample == 0 ? 0 : sample == 1 ? tiles / 2 + 113 : tiles - 1, cta = tile / 256;
    int row, col;
    MmaLaneTile128x128::thread_cell_global(int(cta / (n / 128)) * 128, int(cta % (n / 128)) * 128,
                                           int(tile % 256), row, col);
    const int r[8] = {0, 1, 2, 3, 16, 17, 18, 19}, s[8] = {0, 1, 2, 3, 32, 33, 34, 35};
    uint32_t x = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            x ^= c[size_t(row + r[i]) * n + col + s[j]];
    out[step * 3 + sample] = x;
}
using LeafGemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, int32_t,
    cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 16, 16>;
struct Frame
{
    std::array<std::unique_ptr<Buffer<int16_t>>, 4> a, b;
    std::array<std::unique_ptr<Buffer<uint32_t>>, 7> p;
    Buffer<int8_t> al, ah, bl, bh;
    Buffer<uint32_t> fa, fb;
    Frame(int m, int n, int k)
        : al(size_t(m) * k), ah(size_t(m) * k), bl(size_t(n) * k), bh(size_t(n) * k), fa(m), fb(n)
    {
        for (int i = 0; i < 4; ++i)
        {
            a[i] = std::make_unique<Buffer<int16_t>>(size_t(m / 2) * (k / 2));
            b[i] = std::make_unique<Buffer<int16_t>>(size_t(n / 2) * (k / 2));
        }
        for (auto &x : p)
            x = std::make_unique<Buffer<uint32_t>>(size_t(m / 2) * (n / 2));
    }
};
class Engine
{
    std::string kind;
    int depth;
    std::vector<std::unique_ptr<Frame>> frames;
    void leaf(View a, View b, Out c, int level)
    {
        auto &f = *frames[level];
        if (kind == "simt_int16" || kind == "pairwise_winograd")
        {
            if (kind == "pairwise_winograd")
                pair_factors<<<(a.rows + b.rows + 255) / 256, 256>>>(a, b, f.fa.p, f.fb.p);
            dim3 grid((b.rows + 15) / 16, (a.rows + 15) / 16), block(16, 16);
            if (kind == "pairwise_winograd")
                simt<true><<<grid, block>>>(a, b, c, f.fa.p, f.fb.p);
            else
                simt<false><<<grid, block>>>(a, b, c, nullptr, nullptr);
            return;
        }
        split<<<(size_t(a.rows) * a.k + 255) / 256, 256>>>(a, f.al.p, f.ah.p);
        split<<<(size_t(b.rows) * b.k + 255) / 256, 256>>>(b, f.bl.p, f.bh.p);
        // Conservative interval bounds avoid data-dependent host downloads.
        // They may require extra passes even when this fixture's actual values fit.
        const bool wide_a = !a.int8(), wide_b = !b.int8();
        auto gemm = [&](int8_t *aa, int8_t *bb, int alpha, int beta)
        {
            typename LeafGemm::Arguments args({a.rows, b.rows, a.k}, {aa, a.k}, {bb, b.k},
                                              {reinterpret_cast<int32_t *>(c.p), c.stride},
                                              {reinterpret_cast<int32_t *>(c.p), c.stride},
                                              {alpha, beta});
            LeafGemm op;
            status(op(args));
            ++gemm_calls;
            leaf_macs += uint64_t(a.rows) * b.rows * a.k;
        };
        gemm(f.al.p, f.bl.p, 1, 0);
        if (wide_a)
            gemm(f.ah.p, f.bl.p, 256, 1);
        if (wide_b)
            gemm(f.al.p, f.bh.p, 256, 1);
        if (wide_a && wide_b)
            gemm(f.ah.p, f.bh.p, 65536, 1);
    }
    void recurse(View a, View b, Out c, int level)
    {
        if (level == depth)
        {
            leaf(a, b, c, level);
            return;
        }
        auto &f = *frames[level];
        int m = a.rows / 2, n = b.rows / 2, k = a.k / 2;
        auto a11 = a.sub(0, 0, m, k), a12 = a.sub(0, k, m, k), a21 = a.sub(m, 0, m, k),
             a22 = a.sub(m, k, m, k);
        auto b11 = b.sub(0, 0, n, k), b12 = b.sub(n, 0, n, k), b21 = b.sub(0, k, n, k),
             b22 = b.sub(n, k, n, k);
        auto combine = [&](View x, View y, int16_t *out, int sign)
        {
            add_operands<<<(size_t(x.rows) * x.k + 255) / 256, 256>>>(x, y, out, sign);
            return View{out,
                        x.rows,
                        x.k,
                        x.k,
                        x.lo + (sign == 1 ? y.lo : -y.hi),
                        x.hi + (sign == 1 ? y.hi : -y.lo)};
        };
        auto s = [&](int i, View x, View y, int sign) { return combine(x, y, f.a[i]->p, sign); };
        auto t = [&](int i, View x, View y, int sign) { return combine(x, y, f.b[i]->p, sign); };
        auto product = [&](int i, View x, View y)
        { recurse(x, y, {f.p[i]->p, m, n, n}, level + 1); };
        bool sw = kind == "strassen_winograd";
        if (!sw)
        {
            product(0, s(0, a11, a22, 1), t(0, b11, b22, 1));
            product(1, s(0, a21, a22, 1), b11);
            product(2, a11, t(0, b12, b22, -1));
            product(3, a22, t(0, b21, b11, -1));
            product(4, s(0, a11, a12, 1), b22);
            product(5, s(0, a21, a11, -1), t(0, b11, b12, 1));
            product(6, s(0, a12, a22, -1), t(0, b21, b22, 1));
        }
        else
        {
            auto s1 = s(0, a21, a22, 1), s2 = s(1, s1, a11, -1), s3 = s(2, a11, a21, -1),
                 s4 = s(3, a12, s2, -1);
            auto t1 = t(0, b12, b11, -1), t2 = t(1, b22, t1, -1), t3 = t(2, b22, b12, -1),
                 t4 = t(3, t2, b21, -1);
            product(0, a11, b11);
            product(1, a12, b21);
            product(2, s4, b22);
            product(3, a22, t4);
            product(4, s1, t1);
            product(5, s2, t2);
            product(6, s3, t3);
        }
        recombine<<<(size_t(m) * n + 255) / 256, 256>>>(f.p[0]->p, f.p[1]->p, f.p[2]->p, f.p[3]->p,
                                                        f.p[4]->p, f.p[5]->p, f.p[6]->p, c, sw);
    }

  public:
    uint64_t gemm_calls = 0, leaf_macs = 0;
    Engine(std::string name, int levels, int m, int n, int k) : kind(name), depth(levels)
    {
        if (depth < 0 || depth > 3 || m % (1 << depth) || n % (1 << depth) || k % (1 << depth))
            throw std::runtime_error("bad recursion dimensions");
        for (int d = 0; d <= depth; ++d)
        {
            frames.push_back(std::make_unique<Frame>(m, n, k));
            m /= 2;
            n /= 2;
            k /= 2;
        }
    }
    void multiply(View a, View b, Out c)
    {
        recurse(a, b, c, 0);
        CUDA(cudaGetLastError());
    }
};
struct Inputs
{
    int m, n, pattern;
    Buffer<int8_t> a8, b8;
    Buffer<int16_t> a16, b16;
    std::vector<int8_t> ha, hb;
    Inputs(int mm, int nn, int p)
        : m(mm), n(nn), pattern(p), a8(size_t(m) * 4096), b8(size_t(n) * 4096), a16(a8.count),
          b16(b8.count), ha(a8.count), hb(b8.count)
    {
        uint32_t seed = 0x1234567;
        for (auto *values : {&ha, &hb})
            for (auto &x : *values)
            {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                x = p == 0 ? 0 : p == 1 ? -128 : int8_t(int(seed & 255) - 128);
            }
        if (p == 3)
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
        CUDA(cudaMemcpy(a8.p, ha.data(), ha.size(), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(b8.p, hb.data(), hb.size(), cudaMemcpyHostToDevice));
        extend<<<(a8.count + 255) / 256, 256>>>(a8.p, a16.p, a8.count);
        extend<<<(b8.count + 255) / 256, 256>>>(b8.p, b16.p, b8.count);
        CUDA(cudaDeviceSynchronize());
    }
    View a(int step) const
    {
        return {a16.p + step * 128,      m, 128, 4096, pattern == 3 ? -127 : -128,
                pattern == 3 ? 126 : 127};
    }
    View b(int step) const
    {
        return {b16.p + step * 128,     n, 128, 4096, pattern == 3 ? -63 : -128,
                pattern == 3 ? 63 : 127};
    }
};
static uint64_t checks = 0;
static void check_equal(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                        const std::string &label)
{
    if (a.size() != b.size())
        throw std::runtime_error(label + " size mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
            throw std::runtime_error(label + " mismatch at " + std::to_string(i));
    checks += a.size();
}
static std::vector<uint32_t> download(const Buffer<uint32_t> &b)
{
    std::vector<uint32_t> h(b.count);
    CUDA(cudaMemcpy(h.data(), b.p, b.count * 4, cudaMemcpyDeviceToHost));
    return h;
}
struct Oracle
{
    std::vector<uint32_t> c, xors, digests;
};
static Oracle oracle(Inputs &in)
{
    size_t plane = size_t(in.m) * in.n, tiles = plane / 64;
    Oracle out;
    out.c.resize(32 * plane);
    out.xors.resize(32 * tiles);
    out.digests.resize(8 * tiles);
    for (int i = 0; i < in.m; ++i)
        for (int j = 0; j < in.n; ++j)
        {
            int64_t sum = 0;
            for (int step = 0; step < 32; ++step)
            {
                for (int k = step * 128; k < (step + 1) * 128; ++k)
                    sum += int64_t(in.ha[size_t(i) * 4096 + k]) * in.hb[size_t(j) * 4096 + k];
                out.c[step * plane + size_t(i) * in.n + j] = uint32_t(sum);
            }
        }
    const int r[8] = {0, 1, 2, 3, 16, 17, 18, 19}, s[8] = {0, 1, 2, 3, 32, 33, 34, 35};
    for (size_t tile = 0; tile < tiles; ++tile)
    {
        size_t cta = tile / 256;
        int row, col;
        MmaLaneTile128x128::thread_cell_global(int(cta / (in.n / 128)) * 128,
                                               int(cta % (in.n / 128)) * 128, int(tile % 256), row,
                                               col);
        uint32_t words[16] = {};
        for (int step = 0; step < 32; ++step)
        {
            uint32_t x = 0;
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    x ^= out.c[step * plane + size_t(row + r[i]) * in.n + col + s[j]];
            out.xors[step * tiles + tile] = x;
            uint32_t w = words[step % 16];
            words[step % 16] = (w << 13 | w >> 19) ^ x;
        }
        uint8_t key[32];
        std::fill(key, key + 32, 37);
        blake3_hasher h;
        blake3_hasher_init_keyed(&h, key);
        blake3_hasher_update(&h, words, 64);
        blake3_hasher_finalize(&h, reinterpret_cast<uint8_t *>(out.digests.data() + tile * 8), 32);
    }
    return out;
}
struct RunState
{
    Inputs &in;
    size_t tiles;
    Buffer<uint32_t> c, delta, words, dump, digests, key, samples;
    Buffer<int> found, rows, cols;
    RunState(Inputs &data, bool verify)
        : in(data), tiles(size_t(in.m) * in.n / 64), c(size_t(in.m) * in.n), delta(c.count),
          words(tiles * 16), dump(tiles * 32), digests(verify ? tiles * 8 : 0), key(8), samples(96),
          found(1), rows(1), cols(1)
    {
        CUDA(cudaMemset(key.p, 37, 32));
    }
    void run(Engine &engine, const Oracle *ref = nullptr, bool sampled = false)
    {
        CUDA(cudaMemsetAsync(c.p, 0, c.count * 4));
        CUDA(cudaMemsetAsync(words.p, 0, words.count * 4));
        CUDA(cudaMemsetAsync(found.p, 0, 4));
        engine.gemm_calls = engine.leaf_macs = 0;
        for (int step = 0; step < 32; ++step)
        {
            engine.multiply(in.a(step), in.b(step), {delta.p, in.m, in.n, in.n});
            accumulate<<<(c.count + 255) / 256, 256>>>(c.p, delta.p, c.count);
            fold_tiles<<<(tiles + 255) / 256, 256>>>(c.p, in.m, in.n, step, words.p,
                                                     ref ? dump.p : nullptr);
            if (sampled)
                capture_samples<<<1, 32>>>(c.p, in.n, tiles, step, samples.p);
            if (ref)
            {
                auto h = download(c);
                check_equal(h,
                            std::vector<uint32_t>(ref->c.begin() + step * c.count,
                                                  ref->c.begin() + (step + 1) * c.count),
                            "prefix C");
            }
        }
        finalize<<<(tiles + 255) / 256, 256>>>(words.p, key.p, tiles, in.n, found.p, rows.p, cols.p,
                                               ref ? digests.p : nullptr);
        CUDA(cudaGetLastError());
        if (ref)
        {
            check_equal(download(dump), ref->xors, "tile XOR");
            check_equal(download(digests), ref->digests, "keyed BLAKE3");
        }
    }
};
static std::vector<uint32_t> sample_reference(Inputs &in)
{
    size_t tiles = size_t(in.m) * in.n / 64;
    std::vector<uint32_t> out(96);
    const int r[8] = {0, 1, 2, 3, 16, 17, 18, 19}, s[8] = {0, 1, 2, 3, 32, 33, 34, 35};
    for (int sample = 0; sample < 3; ++sample)
    {
        size_t tile = sample == 0 ? 0 : sample == 1 ? tiles / 2 + 113 : tiles - 1, cta = tile / 256;
        int row, col;
        MmaLaneTile128x128::thread_cell_global(int(cta / (in.n / 128)) * 128,
                                               int(cta % (in.n / 128)) * 128, int(tile % 256), row,
                                               col);
        int64_t c[8][8] = {};
        for (int step = 0; step < 32; ++step)
        {
            uint32_t x = 0;
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                {
                    for (int k = step * 128; k < (step + 1) * 128; ++k)
                        c[i][j] += int64_t(in.ha[size_t(row + r[i]) * 4096 + k]) *
                                   in.hb[size_t(col + s[j]) * 4096 + k];
                    x ^= uint32_t(c[i][j]);
                }
            out[step * 3 + sample] = x;
        }
    }
    return out;
}
static void baseline_samples(RunState &state, const std::vector<uint32_t> &expected)
{
    std::vector<uint32_t> out(96);
    for (int step = 0; step < 32; ++step)
        for (int sample = 0; sample < 3; ++sample)
        {
            size_t tile = sample == 0 ? 0 : sample == 1 ? state.tiles / 2 + 113 : state.tiles - 1;
            CUDA(cudaMemcpy(&out[step * 3 + sample], state.dump.p + step * state.tiles + tile, 4,
                            cudaMemcpyDeviceToHost));
        }
    check_equal(out, expected, "large current CUTLASS samples");
}
template <class T> static void baseline(Inputs &in, RunState &state, bool dump)
{
    CpCutlassJackpotLaunch jp{};
    jp.d_a_key8 = state.key.p;
    jp.d_found = state.found.p;
    jp.d_out_t_rows = state.rows.p;
    jp.d_out_t_cols = state.cols.p;
    CUDA(cudaMemsetAsync(state.found.p, 0, 4));
    cp_cutlass::FusedMilestoneGemmOp<T> op;
    status(op.initialize(in.m, in.n, 4096, in.m, in.n, in.a8.p, in.b8.p,
                         dump ? state.dump.p : nullptr, in.n / 128, state.tiles, &jp));
    status(op());
}
static void measure(const std::string &name, int depth, Inputs &in, RunState &state,
                    const std::function<void()> &run, Engine *engine, int repeats)
{
    for (int i = 0; i < 5; ++i)
        run();
    CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&end));
    std::vector<float> times;
    std::vector<double> wall;
    CUDA(cudaEventRecord(start));
    run();
    CUDA(cudaEventRecord(end));
    CUDA(cudaEventSynchronize(end));
    float pilot;
    CUDA(cudaEventElapsedTime(&pilot, start, end));
    int batch = std::max(1, std::min(128, int(20 / std::max(pilot, 0.001f)) + 1));
    for (int i = 0; i < repeats; ++i)
    {
        auto t = std::chrono::steady_clock::now();
        CUDA(cudaEventRecord(start));
        for (int b = 0; b < batch; ++b)
            run();
        CUDA(cudaEventRecord(end));
        CUDA(cudaEventSynchronize(end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, start, end));
        times.push_back(ms / batch);
        wall.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
                .count() /
            batch);
    }
    int hits;
    CUDA(cudaMemcpy(&hits, state.found.p, 4, cudaMemcpyDeviceToHost));
    if (hits)
        throw std::runtime_error("unexpected zero-target hit");
    std::printf(
        "{\"type\":\"measurement\",\"method\":\"%s\",\"depth\":%d,\"m\":%d,\"n\":%d,\"k\":4096,"
        "\"pattern\":%d,\"int8_gemm_calls\":%llu,\"tensor_leaf_macs\":%llu,\"gpu_ms\":[",
        name.c_str(), depth, in.m, in.n, in.pattern,
        (unsigned long long)(engine ? engine->gemm_calls : 1),
        (unsigned long long)(engine ? engine->leaf_macs : uint64_t(in.m) * in.n * 4096));
    for (size_t i = 0; i < times.size(); ++i)
        std::printf("%s%.6f", i ? "," : "", times[i]);
    std::printf("],\"iterations_per_sample\":%d,\"wall_ms\":[", batch);
    for (size_t i = 0; i < wall.size(); ++i)
        std::printf("%s%.6f", i ? "," : "", wall[i]);
    std::puts("]}");
    std::fflush(stdout);
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(end));
}
int main(int argc, char **argv)
{
    try
    {
        bool verify = argc == 2 && std::string(argv[1]) == "--verify";
        int m = verify ? 256 : (argc > 1 ? std::stoi(argv[1]) : 1024),
            n = verify ? 128 : (argc > 2 ? std::stoi(argv[2]) : 8192),
            repeats = argc > 3 ? std::stoi(argv[3]) : 9;
        if (m < 128 || m > 4096 || n < 128 || n > 131072 || m % 128 || n % 128 || repeats < 3 ||
            repeats > 31)
            throw std::runtime_error("invalid benchmark dimensions/repeats");
        cudaDeviceProp prop;
        CUDA(cudaGetDeviceProperties(&prop, 0));
        if (prop.major * 10 + prop.minor < 75)
            throw std::runtime_error("build requires sm_75 tensor INT8 support");
        std::printf("{\"type\":\"metadata\",\"gpu\":\"%s\",\"compute\":\"%d.%d\",\"verify\":%s}\n",
                    prop.name, prop.major, prop.minor, verify ? "true" : "false");
        std::fflush(stdout);
        for (int pattern : verify ? std::vector<int>{0, 1, 2, 3} : std::vector<int>{3})
        {
            Inputs in(m, n, pattern);
            RunState state(in, verify);
            Oracle ref;
            if (verify)
                ref = oracle(in);
            auto expected_samples = sample_reference(in);
            baseline<cp_cutlass::Gemm128x128TensorOp>(in, state, true);
            CUDA(cudaDeviceSynchronize());
            baseline_samples(state, expected_samples);
            baseline<cp_cutlass::Gemm256x128TensorOp>(in, state, true);
            CUDA(cudaDeviceSynchronize());
            baseline_samples(state, expected_samples);
            baseline<cp_cutlass::Gemm128x128RowMajor>(in, state, true);
            CUDA(cudaDeviceSynchronize());
            baseline_samples(state, expected_samples);
            if (verify)
            {
                check_equal(download(state.dump), ref.xors, "current DP4A XOR");
                baseline<cp_cutlass::Gemm128x128TensorOp>(in, state, true);
                CUDA(cudaDeviceSynchronize());
                check_equal(download(state.dump), ref.xors, "current tensor 128 XOR");
                baseline<cp_cutlass::Gemm256x128TensorOp>(in, state, true);
                CUDA(cudaDeviceSynchronize());
                check_equal(download(state.dump), ref.xors, "current tensor 256 XOR");
            }
            else
            {
                measure(
                    "current_tensor_256", 0, in, state,
                    [&] { baseline<cp_cutlass::Gemm256x128TensorOp>(in, state, false); }, nullptr,
                    repeats);
                measure(
                    "current_tensor_128", 0, in, state,
                    [&] { baseline<cp_cutlass::Gemm128x128TensorOp>(in, state, false); }, nullptr,
                    repeats);
                measure(
                    "current_dp4a", 0, in, state,
                    [&] { baseline<cp_cutlass::Gemm128x128RowMajor>(in, state, false); }, nullptr,
                    repeats);
            }
            for (auto kind : {"tensor_limb_classical", "simt_int16", "pairwise_winograd",
                              "strassen", "strassen_winograd"})
            {
                const bool wide_panel = size_t(m) * n > size_t(4096) * 8192;
                if (wide_panel &&
                    (std::string(kind) == "simt_int16" || std::string(kind) == "pairwise_winograd"))
                {
                    std::printf("{\"type\":\"skipped\",\"method\":\"%s\",\"reason\":\"slow SIMT "
                                "controls measured on smaller panels\"}\n",
                                kind);
                    continue;
                }
                for (int depth = (std::string(kind).find("strassen") == 0 ? 1 : 0);
                     depth <= (std::string(kind).find("strassen") == 0 ? (wide_panel ? 1 : 3) : 0);
                     ++depth)
                {
                    Engine engine(kind, depth, m, n, 128);
                    state.run(engine, nullptr, true);
                    CUDA(cudaDeviceSynchronize());
                    check_equal(download(state.samples), expected_samples,
                                "large prototype samples");
                    if (verify)
                    {
                        state.run(engine, &ref);
                        CUDA(cudaDeviceSynchronize());
                        std::printf("{\"type\":\"correctness\",\"method\":\"%s\",\"depth\":%d,"
                                    "\"pattern\":%d,\"passed\":true}\n",
                                    kind, depth, pattern);
                        std::fflush(stdout);
                    }
                    else
                        measure(
                            kind, depth, in, state, [&] { state.run(engine); }, &engine, repeats);
                }
            }
        }
        std::printf("{\"type\":\"complete\",\"passed\":true,\"checked_values\":%llu}\n",
                    (unsigned long long)checks);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
