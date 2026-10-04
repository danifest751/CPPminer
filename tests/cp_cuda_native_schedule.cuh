// Load an isolated sm75 cubin through the CUDA driver API. Host params and proof
// layout remain the same as the runtime kernel used by the independent oracle.
#pragma once
#include <cuda.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace cp_research {
inline void driver_check(CUresult result) {
    if (result!=CUDA_SUCCESS) {
        const char *message=nullptr;cuGetErrorString(result,&message);
        throw std::runtime_error(std::string("native cubin: ")+(message?message:"driver error"));
    }
}
inline CUmodule native_module() {
    static CUmodule module=nullptr;
    if (!module) {
        if (cudaFree(nullptr)!=cudaSuccess) throw std::runtime_error("CUDA context init failed");
        const char *path=std::getenv("CP_RESEARCH_CUBIN");
        if (!path) throw std::runtime_error("CP_RESEARCH_CUBIN is required");
        driver_check(cuModuleLoad(&module,path));
    }
    return module;
}
inline void native_symbol(const char *name,const void *data,size_t size) {
    CUdeviceptr pointer;size_t bytes;
    driver_check(cuModuleGetGlobal(&pointer,&bytes,native_module(),name));
    if (bytes!=size) throw std::runtime_error("native symbol size mismatch");
    driver_check(cuMemcpyHtoD(pointer,data,size));
}
template<class T> struct NativeOp {
    cp_cutlass::FusedMilestoneGemmOp<T> original;
    CUfunction function=nullptr;
    template<class... Args> cutlass::Status initialize(Args&&... args) {
        auto status=original.initialize(std::forward<Args>(args)...);
        if (status!=cutlass::Status::kSuccess) return status;
        const char *name=std::getenv(std::is_same<T,cp_cutlass::Gemm256x128TensorOp>::value
                                   ? "CP_RESEARCH_FUNCTION_256" : "CP_RESEARCH_FUNCTION_128");
        if (!name) throw std::runtime_error("native kernel symbol is required");
        driver_check(cuModuleGetFunction(&function,native_module(),name));
        const int shared=sizeof(typename T::GemmKernel::SharedStorage);
        if (shared>=48*1024)
            driver_check(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
        return status;
    }
    cutlass::Status operator()() {
        auto grid=typename T::ThreadblockSwizzle().get_grid_shape(original.params.grid_tiled_shape);
        void *arguments[]={&original.params};
        driver_check(cuLaunchKernel(function,grid.x,grid.y,grid.z,T::GemmKernel::kThreadCount,1,1,
                     sizeof(typename T::GemmKernel::SharedStorage),nullptr,arguments,nullptr));
        return cutlass::Status::kSuccess;
    }
};
} // namespace cp_research
