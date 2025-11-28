#pragma once
#include <cuda.h>
#include <map>
#include <mutex>

// O(log N)のBest-Fitアロケータ
class VmmAllocator {
public:
    // 初期化: 巨大な仮想アドレス空間を登録
    void init(CUdeviceptr base, size_t size);
    
    // 割当: サイズに合う最小の空きブロックを返す (Best-Fit)
    CUdeviceptr alloc(size_t size);
    
    // 解放: ブロックを戻し、隣接ブロックと結合(Coalescing)する
    void free(CUdeviceptr ptr, size_t size);
    
    size_t get_total_free() const;

private:
    std::mutex mtx_;
    size_t total_free_ = 0;

    // Dual-Map構造
    // 1. サイズ順: Best-Fit検索用 (Size -> Ptr)
    //    同じサイズが複数あるため multimap
    std::multimap<size_t, CUdeviceptr> free_by_size_;

    // 2. アドレス順: Coalescing用 (Ptr -> Size)
    //    隣接チェック用
    std::map<CUdeviceptr, size_t> free_by_addr_;
};