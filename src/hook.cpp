#include <cuda_runtime.h>
#include <dlfcn.h>
#include "../include/vmm_manager.hpp"
#include "../include/vmm_env.hpp"
#include "../include/vmm_logger.hpp"

typedef cudaError_t (*real_cudaMalloc_t)(void**, size_t);
typedef cudaError_t (*real_cudaFree_t)(void*);
typedef cudaError_t (*real_cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);

static real_cudaMalloc_t real_cudaMalloc = nullptr;
static real_cudaFree_t real_cudaFree = nullptr;
static real_cudaMemcpy_t real_cudaMemcpy = nullptr;

static void load_symbols() {
    if (!real_cudaMalloc) real_cudaMalloc = (real_cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    if (!real_cudaFree) real_cudaFree = (real_cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cudaMemcpy) real_cudaMemcpy = (real_cudaMemcpy_t)dlsym(RTLD_NEXT, "cudaMemcpy");
}

extern "C" {
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    load_symbols();
    if (VmmEnv::get().get_mode() == VmmMode::MONITOR) {
        cudaError_t err = real_cudaMalloc(devPtr, size);
        if (err == cudaSuccess) LOG_INFO("ALLOC", *devPtr, size, "MONITOR");
        return err;
    }
    return VmmManager::get().allocate(devPtr, size);
}

cudaError_t cudaFree(void* devPtr) {
    load_symbols();
    if (VmmEnv::get().get_mode() == VmmMode::MONITOR) {
        LOG_INFO("FREE", devPtr, 0, "MONITOR");
        return real_cudaFree(devPtr);
    }
    return VmmManager::get().free(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    load_symbols();
    if (VmmEnv::get().get_mode() == VmmMode::VMM && kind == cudaMemcpyDeviceToDevice) {
        if (VmmManager::get().try_optimize_memcpy(dst, src, count)) return cudaSuccess;
    }
    cudaError_t err = real_cudaMemcpy(dst, src, count, kind);
    if (VmmEnv::get().get_mode() == VmmMode::MONITOR && err == cudaSuccess) {
        LOG_INFO("COPY", dst, count, "MONITOR");
    }
    return err;
}
}