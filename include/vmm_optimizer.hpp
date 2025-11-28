#pragma once
#include <cuda_runtime.h>
#include "vmm_driver_wrapper.hpp"
#include "vmm_tracker.hpp"
#include "vmm_allocator.hpp"
#include "vmm_pool.hpp" // Pool追加

class VmmOptimizer {
public:
    VmmOptimizer(VmmDriverWrapper& drv, VmmTracker& trk, VmmAllocator& alloc, VmmPhysicalPool& pool);
    bool try_optimize_copy(void* dst, const void* src, size_t count);

private:
    VmmDriverWrapper& driver_;
    VmmTracker& tracker_;
    VmmAllocator& allocator_;
    VmmPhysicalPool& pool_; // Pool追加
};