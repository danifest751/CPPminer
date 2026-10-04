// Register-only DPAS throughput on an Intel GPU (OpenCL, SIMD8 subgroups).
// Each work item issues 8 independent intel_sub_group_i8_i8_matrix_mad_k32
// per iteration; with SG8 and an int8 A operand one call is an 8x8x32 int8
// MAC block per subgroup (2048 MAC). (short8 A is the SG16 overload.)
// The result is the XMX ceiling a GEMM kernel could approach on this device.
//
//   g++ -O2 -o dpas_peak_bench dpas_peak_bench.cpp -lOpenCL && ./dpas_peak_bench
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <vector>

static const char *kSrc = R"CLC(
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void dpas_peak(__global int *out, int iters, int seed) {
    /* Distinct A operands per accumulator so no two calls can be merged; B
     * changes every iteration so nothing can be hoisted out of the loop. */
    const int base = seed + (int)get_global_id(0);
    const int8 a0 = (int8)(base), a1 = (int8)(base + 1), a2 = (int8)(base + 2),
               a3 = (int8)(base + 3), a4 = (int8)(base + 4), a5 = (int8)(base + 5),
               a6 = (int8)(base + 6), a7 = (int8)(base + 7);
    int8 b = (int8)(seed ^ (int)get_global_id(0));
    int8 c0 = 0, c1 = 1, c2 = 2, c3 = 3, c4 = 4, c5 = 5, c6 = 6, c7 = 7;
    for (int i = 0; i < iters; ++i) {
        c0 = intel_sub_group_i8_i8_matrix_mad_k32(a0, b, c0);
        c1 = intel_sub_group_i8_i8_matrix_mad_k32(a1, b, c1);
        c2 = intel_sub_group_i8_i8_matrix_mad_k32(a2, b, c2);
        c3 = intel_sub_group_i8_i8_matrix_mad_k32(a3, b, c3);
        c4 = intel_sub_group_i8_i8_matrix_mad_k32(a4, b, c4);
        c5 = intel_sub_group_i8_i8_matrix_mad_k32(a5, b, c5);
        c6 = intel_sub_group_i8_i8_matrix_mad_k32(a6, b, c6);
        c7 = intel_sub_group_i8_i8_matrix_mad_k32(a7, b, c7);
        b = b + (int8)(i);
    }
    int8 s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out[get_global_id(0)] = s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7;
}
)CLC";

int main() {
    cl_platform_id plats[8];
    cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_device_id dev = nullptr;
    for (cl_uint p = 0; p < np && !dev; ++p) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &dev, &nd) != CL_SUCCESS) dev = nullptr;
    }
    if (!dev) { std::puts("no GPU"); return 1; }
    char name[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kSrc, nullptr, &err);
    if (clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr) != CL_SUCCESS) {
        char log[8192] = {};
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::printf("build failed:\n%s\n", log);
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "dpas_peak", &err);
    const size_t local = 64;
    for (size_t global : {size_t(16384), size_t(65536), size_t(262144)}) {
        cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, global * sizeof(int), nullptr, &err);
        const int iters = 4096, seed = 3;
        clSetKernelArg(k, 0, sizeof(cl_mem), &out);
        clSetKernelArg(k, 1, sizeof(int), &iters);
        clSetKernelArg(k, 2, sizeof(int), &seed);
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        const cl_int fin = clFinish(q); // warm-up
        if (err != CL_SUCCESS || fin != CL_SUCCESS) {
            std::printf("launch failed: enqueue %d finish %d\n", err, fin);
            return 1;
        }
        const int reps = 5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        clFinish(q);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double subgroups = double(global) / 8.0;
        const double macs = subgroups * iters * 8.0 /*dpas per iter*/ * 2048.0 * reps;
        std::printf("%s global=%zu: %.2f TMAC/s\n", name, global, macs / sec / 1e12);
        clReleaseMemObject(out);
    }
    return 0;
}
