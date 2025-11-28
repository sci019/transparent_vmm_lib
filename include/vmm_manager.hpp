#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include "vmm_driver_wrapper.hpp"
#include "vmm_tracker.hpp"
#include "vmm_optimizer.hpp"
#include "vmm_allocator.hpp"
#include "vmm_pool.hpp" // Pool追加

class VmmManager {
public:
    static VmmManager& get();
    void init();
    cudaError_t allocate(void** dptr, size_t size);
    cudaError_t free(void* dptr);
    bool try_optimize_memcpy(void* dst, const void* src, size_t count);

    // Optimizerからプール操作を行うため公開
    VmmPhysicalPool& get_pool() { return pool_; }

private:
    VmmManager(); 
    ~VmmManager() = default;
    
    std::once_flag init_flag_;
    size_t granularity_ = 0;
    
    CUdeviceptr pool_base_ = 0;
    size_t pool_size_ = 0;

    VmmDriverWrapper driver_;
    VmmTracker tracker_;
    VmmAllocator allocator_;
    VmmPhysicalPool pool_; // Poolインスタンス
    VmmOptimizer optimizer_;
};