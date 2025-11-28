#pragma once
#include <cuda.h>

class VmmDriverWrapper {
public:
    void init();
    size_t get_granularity();
    size_t get_total_memory(); // 新規: 物理容量取得用
    CUdeviceptr reserve_address(size_t size);
    void free_address(CUdeviceptr ptr, size_t size);
    CUmemGenericAllocationHandle create_physical_mem(size_t size);
    void release_physical_mem(CUmemGenericAllocationHandle handle);
    void map_memory(CUdeviceptr ptr, size_t size, CUmemGenericAllocationHandle handle);
    void unmap_memory(CUdeviceptr ptr, size_t size);
};