// TNet v1 on OpenCL (AMD, Intel and other GPUs without CUDA): the host side of kernels/tnet.cl. The int8 GEMM runs on the
// GPU's integer dot product (cl_khr_integer_dot_product where the driver has it) with the requantization in its
// epilogue; expansion and tickets are SHA-256 kernels. The network and the mining loop are cp_tnet_net.cpp.

#include "cp_tnet.h"

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL

#include "cp_tnet_common.hpp"
#include "../opencl/case33_gemm_ocl.hpp"
#include "../opencl/opencl_context.hpp"

#include <chrono>
#include <cmath>
#include <random>

namespace {

std::string tnet_kernel_path() {
    // next to the other OpenCL kernels (kernels/ beside the executable)
    std::string p = cp_ocl_resolve_kernel_path();
    const size_t slash = p.find_last_of("\\/");
    return (slash == std::string::npos ? std::string() : p.substr(0, slash + 1)) + "tnet.cl";
}

struct RqOcl : RqDevice {
    int index = 0;
    OpenClContext ctx;
    int dot = 0;  // RQ_DOT of the program in use (tnet.cl)
    bool error = false, ready = false;
    cl_kernel k_expand = nullptr, k_transpose = nullptr, k_gemm = nullptr, k_tickets = nullptr;
    std::vector<cl_mem> WT;
    cl_mem tmp = nullptr, X = nullptr, X2 = nullptr, dfound = nullptr, dhits = nullptr;
    int n = 0, L = 0, w = 0, mult = 0, batch = 0;
    std::vector<uint8_t> seed;

    explicit RqOcl(int device) : index(device) {}
    ~RqOcl() override {
        for (cl_mem m : WT) if (m) clReleaseMemObject(m);
        for (cl_mem m : {tmp, X, X2, dfound, dhits}) if (m) clReleaseMemObject(m);
        for (cl_kernel k : {k_expand, k_transpose, k_gemm, k_tickets}) if (k) clReleaseKernel(k);
    }

    std::string name() const override { return ctx.device_name; }
    std::string kernel() const override {
        static const char* how[] = {"multiply-add", "dot product (cl_khr_integer_dot_product)", "AMD sudot4 (v_dot4_i32_iu8)",
                                    "AMD sdot4 (v_dot4_i32_i8)"};
        return std::string("OpenCL int8 ") + how[dot] + ", 64x64 tiles";
    }
    bool failed() override {
        const bool e = error;
        error = false;
        return e;
    }
    void check(cl_int e) { if (e != CL_SUCCESS) error = true; }

    void release_program() {
        for (cl_kernel* k : {&k_expand, &k_transpose, &k_gemm, &k_tickets}) if (*k) { clReleaseKernel(*k); *k = nullptr; }
        if (ctx.program) { clReleaseProgram(ctx.program); ctx.program = nullptr; }
    }

    // context, program and kernels: the ways to sum int8 products in turn (AMD's dot instructions, the KHR dot
    // product, then written out), keeping the first that builds and agrees with the CPU; CP_TNET_OCL_DOT=0..3 forces one
    bool open() {
        if (ready) return true;
        if (!ctx.init(index, -1)) { fprintf(stderr, "[tnet] no OpenCL device %d (cppminer --list-devices)\n", index); return false; }
        const std::string path = tnet_kernel_path();
        const bool amd = ctx.vendor_name.find("Advanced Micro") != std::string::npos || ctx.device_name.compare(0, 3, "gfx") == 0;
        std::vector<int> order;
        if (amd) order = {2, 3};
        if (ctx.has_integer_dot_product) order.push_back(1);
        order.push_back(0);
        if (const char* e = getenv("CP_TNET_OCL_DOT")) order = {atoi(e) & 3};
        for (int d : order) {
            release_program();
            // the KHR dot product is an OpenCL C 3.0 built-in
            const std::string opts = "-DRQ_DOT=" + std::to_string(d) + (d == 1 ? " -cl-std=CL3.0" : "");
            if (!ctx.safe_build_program_from_file(path.c_str(), opts.c_str(), d != 0)) continue;
            k_expand = ctx.create_kernel("rq_expand");
            k_transpose = ctx.create_kernel("rq_transpose");
            k_gemm = ctx.create_kernel("rq_gemm_rq");
            k_tickets = ctx.create_kernel("rq_tickets");
            if (!(k_expand && k_transpose && k_gemm && k_tickets)) continue;
            dot = d;
            if (gemm_ok(130, 256, 2505)) { ready = true; return true; }
        }
        release_program();
        fprintf(stderr, "[tnet] no OpenCL kernel of %s builds and agrees with the CPU\n", path.c_str());
        return false;
    }

    // the requantizing GEMM against the CPU (rows not a multiple of the tile)
    bool gemm_ok(int rows, int cols, int m) {
        std::mt19937 rng(54321 + m);
        std::vector<int8_t> x((size_t)rows * cols), wt((size_t)cols * cols), xo((size_t)rows * cols);
        for (auto& v : x) v = (int8_t)(rng() & 0xff);
        for (auto& v : wt) v = (int8_t)(rng() & 0xff);
        cl_mem dx = ctx.alloc_buffer(x.size(), CL_MEM_READ_ONLY), dw = ctx.alloc_buffer(wt.size(), CL_MEM_READ_ONLY),
               dxo = ctx.alloc_buffer(xo.size(), CL_MEM_READ_WRITE);
        bool ok = dx && dw && dxo && ctx.write_buffer(dx, x.data(), x.size()) && ctx.write_buffer(dw, wt.data(), wt.size());
        if (ok) {
            const int keep = mult;
            mult = m;
            gemm_rq(dx, dw, dxo, rows, cols);
            mult = keep;
            ok = ctx.read_buffer(dxo, xo.data(), xo.size()) && !failed() && xo == rq_cpu_gemm_rq(x, wt, rows, cols, m);
        }
        for (cl_mem b : {dx, dw, dxo}) if (b) clReleaseMemObject(b);
        return ok;
    }

    void expand(cl_mem out, const uint8_t s[32], uint32_t c0, size_t hashes) {
        RqPrefix px = rq_prefix(s);
        cl_ulong count = hashes;
        check(clSetKernelArg(k_expand, 0, sizeof px, &px));
        check(clSetKernelArg(k_expand, 1, sizeof c0, &c0));
        check(clSetKernelArg(k_expand, 2, sizeof count, &count));
        check(clSetKernelArg(k_expand, 3, sizeof out, &out));
        const size_t local = 256, global = (hashes + local - 1) / local * local;
        check(clEnqueueNDRangeKernel(ctx.queue, k_expand, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    }

    void gemm_rq(cl_mem x, cl_mem wt, cl_mem xo, int rows, int cols) {
        check(clSetKernelArg(k_gemm, 0, sizeof x, &x));
        check(clSetKernelArg(k_gemm, 1, sizeof wt, &wt));
        check(clSetKernelArg(k_gemm, 2, sizeof xo, &xo));
        check(clSetKernelArg(k_gemm, 3, sizeof rows, &rows));
        check(clSetKernelArg(k_gemm, 4, sizeof cols, &cols));
        check(clSetKernelArg(k_gemm, 5, sizeof mult, &mult));
        const size_t local[2] = {16, 16};
        const size_t global[2] = {(size_t)cols / 64 * 16, (size_t)(rows + 63) / 64 * 16};
        check(clEnqueueNDRangeKernel(ctx.queue, k_gemm, 2, nullptr, global, local, 0, nullptr, nullptr));
    }

    void tickets(cl_mem x, int rows, int cols, int width, uint32_t r0, const RqTail& tail, const RqTarget& tg, cl_mem found,
                 cl_mem hits) {
        check(clSetKernelArg(k_tickets, 0, sizeof x, &x));
        check(clSetKernelArg(k_tickets, 1, sizeof cols, &cols));
        check(clSetKernelArg(k_tickets, 2, sizeof rows, &rows));
        check(clSetKernelArg(k_tickets, 3, sizeof width, &width));
        check(clSetKernelArg(k_tickets, 4, sizeof r0, &r0));
        check(clSetKernelArg(k_tickets, 5, sizeof tail, &tail));
        check(clSetKernelArg(k_tickets, 6, sizeof tg, &tg));
        check(clSetKernelArg(k_tickets, 7, sizeof found, &found));
        check(clSetKernelArg(k_tickets, 8, sizeof hits, &hits));
        const size_t count = (size_t)rows * (cols / width), local = 64, global = (count + local - 1) / local * local;
        check(clEnqueueNDRangeKernel(ctx.queue, k_tickets, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    }

    bool init(const RqWork& wk, int rows) override {
        n = wk.n; L = wk.L; w = wk.w; mult = wk.mult; batch = rows;
        if (n % 64 || w % 64 || n % w) { fprintf(stderr, "[tnet] unsupported parameters n=%d w=%d\n", n, w); return false; }
        if (!open()) return false;
        if (fused == 0) {
            fprintf(stderr, "[tnet] OpenCL requantizes in the GEMM only\n");
            fused = 1;
            return true;  // the benchmark skips the mode it asked for
        }
        fused = 1;
        const size_t nn = (size_t)n * n, bn = (size_t)batch * n;
        WT.assign(L, nullptr);
        for (auto& p : WT) if (!(p = ctx.alloc_buffer(nn, CL_MEM_READ_ONLY))) return false;
        tmp = ctx.alloc_buffer((nn + 31) / 32 * 32, CL_MEM_READ_WRITE);
        X = ctx.alloc_buffer((bn + 31) / 32 * 32 + 32, CL_MEM_READ_WRITE);
        X2 = ctx.alloc_buffer(bn, CL_MEM_READ_WRITE);
        dfound = ctx.alloc_buffer(4, CL_MEM_READ_WRITE);
        dhits = ctx.alloc_buffer(8 * RQ_MAX_HITS, CL_MEM_READ_WRITE);
        return tmp && X && X2 && dfound && dhits;
    }

    // W_l = expand(SHA256("abacus/tnet-w" || seed || LE32(l)), n^2), stored transposed.
    void set_epoch(const std::vector<uint8_t>& s) override {
        if (s == seed) return;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t nn = (size_t)n * n, hashes = (nn + 31) / 32;
        for (int l = 0; l < L; ++l) {
            uint8_t sl[32];
            rq_w_seed(s, l, sl);
            expand(tmp, sl, 0, hashes);
            check(clSetKernelArg(k_transpose, 0, sizeof tmp, &tmp));
            check(clSetKernelArg(k_transpose, 1, sizeof(cl_mem), &WT[l]));
            check(clSetKernelArg(k_transpose, 2, sizeof n, &n));
            const size_t local[2] = {32, 8}, global[2] = {(size_t)n, (size_t)n / 4};
            check(clEnqueueNDRangeKernel(ctx.queue, k_transpose, 2, nullptr, global, local, 0, nullptr, nullptr));
        }
        check(clFinish(ctx.queue));
        seed = s;
        printf("[tnet] epoch weights ready (%.2f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    void scan(const RqWork& wk, uint64_t nonce, uint32_t r0, int rows, std::vector<RqHit>& hits) override {
        uint8_t xs[32];
        rq_x0_seed(wk.digest, nonce, xs);
        const size_t bn = (size_t)rows * n, hashes = (bn + 31) / 32;
        const uint32_t c0 = (uint32_t)((size_t)r0 * n / 32);
        const cl_uint zero = 0;
        check(clEnqueueWriteBuffer(ctx.queue, dfound, CL_FALSE, 0, 4, &zero, 0, nullptr, nullptr));
        auto t0 = std::chrono::steady_clock::now();
        auto lap = [&](double& acc) {
            if (!profile) return;
            check(clFinish(ctx.queue));
            const auto t1 = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
            t0 = t1;
        };
        expand(X, xs, c0, hashes);
        lap(ms_expand);
        cl_mem cur = X, nxt = X2;
        for (int l = 0; l < L; ++l) {
            gemm_rq(cur, WT[l], nxt, rows, n);
            std::swap(cur, nxt);
        }
        lap(ms_layers);
        tickets(cur, rows, n, w, r0, rq_tail(wk.digest.data(), nonce, w), rq_target(wk.target), dfound, dhits);
        cl_uint found = 0;
        check(clEnqueueReadBuffer(ctx.queue, dfound, CL_TRUE, 0, 4, &found, 0, nullptr, nullptr));
        lap(ms_hash);
        if (profile) ++scans;
        if (found == 0) return;
        const unsigned k = found < RQ_MAX_HITS ? found : RQ_MAX_HITS;
        cl_ulong packed[RQ_MAX_HITS];
        check(clEnqueueReadBuffer(ctx.queue, dhits, CL_TRUE, 0, 8 * k, packed, 0, nullptr, nullptr));
        for (unsigned q = 0; q < k; ++q) {
            const uint32_t i = (uint32_t)(packed[q] >> 32);
            RqHit h;
            h.nonce = nonce;
            h.row = r0 + i;
            h.c = (uint32_t)(packed[q] & 0xFFFFFFFF);
            h.piece.resize(w);
            check(clEnqueueReadBuffer(ctx.queue, cur, CL_TRUE, (size_t)i * n + (size_t)h.c * w, w, h.piece.data(), 0, nullptr, nullptr));
            hits.push_back(std::move(h));
        }
    }
};

}  // namespace

extern "C" int cp_tnet_ocl_selftest(int device) {
    RqOcl d(device);
    if (!d.open()) return 1;
    printf("[tnet] selftest on %s (%s)\n", d.ctx.device_name.c_str(), d.kernel().c_str());
    // expansion blocks 5 and 6 of seed 0x03..
    uint8_t seed[32];
    memset(seed, 3, 32);
    cl_mem out = d.ctx.alloc_buffer(64, CL_MEM_READ_WRITE);
    d.expand(out, seed, 5, 2);
    uint8_t got[64];
    d.ctx.read_buffer(out, got, 64);
    clReleaseMemObject(out);
    const bool exp = memcmp(got, rq_cpu_expand(seed, 5, 2).data(), 64) == 0;
    // ticket hashing against a target equal to one ticket's hash (it must be found) and one below it (it must not)
    bool tk = true;
    {
        const int tn = 512, tw = 256, trows = 4;
        const uint32_t r0 = 70000, ti = 3, tc = 1;
        const uint64_t tnonce = 0x0123456789abcdefULL;
        std::vector<int8_t> x((size_t)trows * tn);
        std::mt19937 rng(99);
        for (auto& v : x) v = (int8_t)(rng() & 0xff);
        uint8_t hd[32];
        for (int q = 0; q < 32; ++q) hd[q] = (uint8_t)(q * 7 + 1);
        uint8_t want[32];
        rq_cpu_ticket(x.data() + ti * tn + tc * tw, tw, hd, tnonce, r0 + ti, tc, want);
        cl_mem dx = d.ctx.alloc_buffer(x.size(), CL_MEM_READ_ONLY), df = d.ctx.alloc_buffer(4, CL_MEM_READ_WRITE),
               dh = d.ctx.alloc_buffer(8 * RQ_MAX_HITS, CL_MEM_READ_WRITE);
        d.ctx.write_buffer(dx, x.data(), x.size());
        for (int below = 0; below < 2; ++below) {
            std::vector<uint8_t> t(want, want + 32);
            RqTarget tg = rq_target(t);
            if (below) tg.t[7] -= 1;  // the hash must not end in 0 (checked below)
            const cl_uint zero = 0;
            d.ctx.write_buffer(df, &zero, 4);
            d.tickets(dx, trows, tn, tw, r0, rq_tail(hd, tnonce, tw), tg, df, dh);
            cl_uint f = 0;
            cl_ulong hits[RQ_MAX_HITS] = {0};
            d.ctx.read_buffer(df, &f, 4);
            d.ctx.read_buffer(dh, hits, sizeof hits);
            bool seen = false;
            for (unsigned q = 0; q < f && q < RQ_MAX_HITS; ++q) seen |= hits[q] == (((cl_ulong)ti << 32) | tc);
            tk = tk && (below ? !seen : seen) && want[31] != 0;
        }
        for (cl_mem m : {dx, df, dh}) clReleaseMemObject(m);
    }
    // the requantizing GEMM against the CPU, rows not a multiple of the tile; two multipliers (both clamps)
    const bool rq = d.gemm_ok(200, 512, 2505) && d.gemm_ok(200, 512, 300000);
    const bool ok = !d.failed();
    printf("[tnet] selftest: expansion %s, tickets %s, requantizing GEMM %s, opencl %s\n", exp ? "ok" : "FAIL", tk ? "ok" : "FAIL",
           rq ? "ok" : "FAIL", ok ? "ok" : "FAIL");
    return exp && tk && rq && ok ? 0 : 1;
}

extern "C" int cp_tnet_ocl_bench(int device, int batch, double seconds) {
    return rq_bench([device] { return std::unique_ptr<RqDevice>(new RqOcl(device)); }, batch, seconds);
}

extern "C" int cp_tnet_ocl_solo(const char* rpc, const char* payee_hex, const char* worker, int device, int batch, double seconds,
                                long long blocks) {
    return rq_solo([device] { return std::unique_ptr<RqDevice>(new RqOcl(device)); }, rpc, payee_hex, worker, batch, seconds,
                   blocks, device);
}

#endif  // CP_ENABLE_OPENCL
