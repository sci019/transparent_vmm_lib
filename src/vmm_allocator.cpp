#include "../include/vmm_allocator.hpp"
#include "../include/vmm_logger.hpp"

void VmmAllocator::init(CUdeviceptr base, size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    free_by_size_.insert({size, base});
    free_by_addr_.insert({base, size});
    total_free_ = size;
}

CUdeviceptr VmmAllocator::alloc(size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    // 1. Best-Fit検索: 要求サイズ以上の最小ブロックを探す O(logN)
    auto it = free_by_size_.lower_bound(size);
    if (it == free_by_size_.end()) {
        LOG_ERR("ALLOC_OOM", nullptr, size, "Allocator OOM (No fit block in VA Pool)");
        return 0; // OOM
    }

    size_t blk_size = it->first;
    CUdeviceptr blk_ptr = it->second;

    // 2. リストから削除 O(logN)
    free_by_size_.erase(it);
    free_by_addr_.erase(blk_ptr);

    // 3. ブロック分割 (余りがあれば戻す)
    if (blk_size > size) {
        size_t rem_size = blk_size - size;
        CUdeviceptr rem_ptr = blk_ptr + size;
        
        free_by_size_.insert({rem_size, rem_ptr});
        free_by_addr_.insert({rem_ptr, rem_size});
    }

    total_free_ -= size;
    return blk_ptr;
}

void VmmAllocator::free(CUdeviceptr ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);

    // 結合対象候補
    CUdeviceptr new_ptr = ptr;
    size_t new_size = size;

    // 1. 前方結合チェック O(logN)
    //    ptrの直前にあるブロックを探す
    //    lower_bound(ptr) は ptr 以上の最初の要素を返すため、その1つ前を見る
    auto it_next = free_by_addr_.lower_bound(ptr);
    if (it_next != free_by_addr_.begin()) {
        auto it_prev = std::prev(it_next);
        if (it_prev->first + it_prev->second == ptr) {
            // 前のブロックと結合可能
            new_ptr = it_prev->first;
            new_size += it_prev->second;
            
            // 古い情報を削除 (イテレータ無効化に注意しつつ)
            // by_sizeからは {size, ptr} のペアを探して消す必要がある
            auto range = free_by_size_.equal_range(it_prev->second);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == it_prev->first) {
                    free_by_size_.erase(it);
                    break;
                }
            }
            free_by_addr_.erase(it_prev);
        }
    }

    // 2. 後方結合チェック O(1) in iterator
    //    it_next は ptr の次にあるブロック
    if (it_next != free_by_addr_.end() && it_next->first == ptr + size) {
        new_size += it_next->second;
        
        auto range = free_by_size_.equal_range(it_next->second);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == it_next->first) {
                free_by_size_.erase(it);
                break;
            }
        }
        free_by_addr_.erase(it_next);
    }

    // 3. 結合後のブロックを挿入 O(logN)
    free_by_size_.insert({new_size, new_ptr});
    free_by_addr_.insert({new_ptr, new_size});
    
    total_free_ += size;
}

size_t VmmAllocator::get_total_free() const {
    return total_free_;
}