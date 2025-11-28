#pragma once
#include <cuda.h>
#include <unordered_map>
#include <vector>
#include <mutex>

struct AllocationInfo {
    CUdeviceptr dptr;
    
    // ★修正: 仮想アドレスサイズを独立して管理
    // mapped_size が 0 になっても（物理メモリが移動しても）、
    // 解放すべき仮想アドレスのサイズは変わらないため。
    size_t va_size;       
    
    size_t mapped_size;     // 現在マップされている物理サイズ
    size_t user_size;       // アプリ要求サイズ
    std::vector<std::pair<CUmemGenericAllocationHandle, size_t>> handles;
};

class VmmTracker {
public:
    void register_alloc(void* ptr, AllocationInfo info);
    void unregister_alloc(void* ptr);
    bool get_alloc(void* ptr, AllocationInfo& out_info);
    void update_alloc(void* ptr, const AllocationInfo& info);

private:
    std::unordered_map<void*, AllocationInfo> map_;
    std::mutex mtx_;
};