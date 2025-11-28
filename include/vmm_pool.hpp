#pragma once
#include <cuda.h>
#include <map>
#include <mutex>
#include <vector>

// 前方宣言
class VmmDriverWrapper;

// 物理メモリハンドルをサイズ別にキャッシュするプール
class VmmPhysicalPool {
public:
    // 指定サイズと「完全に一致する」ハンドルを取得 (なければ0)
    CUmemGenericAllocationHandle pop(size_t size);

    // 指定サイズ「以下」で「最大」のハンドルを取得
    bool pop_largest_le(size_t req_size, CUmemGenericAllocationHandle* out_handle, size_t* out_size);
    
    // 物理ハンドルをプールに返却
    void push(size_t size, CUmemGenericAllocationHandle handle);

    // ★追加: プール内の全ハンドルを解放する (緊急用)
    size_t drain(VmmDriverWrapper& driver);

private:
    std::mutex mtx_;
    // size -> handles
    std::multimap<size_t, CUmemGenericAllocationHandle> pool_;
};