#include "../include/vmm_optimizer.hpp"
#include "../include/vmm_logger.hpp"
#include <iostream>

VmmOptimizer::VmmOptimizer(VmmDriverWrapper& drv, VmmTracker& trk, VmmAllocator& alloc, VmmPhysicalPool& pool) 
    : driver_(drv), tracker_(trk), allocator_(alloc), pool_(pool) {}

bool VmmOptimizer::try_optimize_copy(void* dst, const void* src, size_t count) {
    AllocationInfo dst_info, src_info;

    if (!tracker_.get_alloc(dst, dst_info) || !tracker_.get_alloc((void*)src, src_info)) {
        return false; 
    }

    if (count != src_info.user_size) return false;
    if (dst_info.user_size < src_info.user_size) return false;

    try {
        // 1. dst (拡張先) の物理メモリをマップ解除
        driver_.unmap_memory(dst_info.dptr, dst_info.mapped_size);
        
        // ★修正: 破棄せずプールへ返却 (ここで確保コストが無駄ではなく「資産」になる)
        for (auto& pair : dst_info.handles) {
            pool_.push(pair.second, pair.first);
        }
        dst_info.handles.clear();

        // 2. src (拡張元) の物理メモリを dst にマップ
        size_t current_offset = 0;
        for (auto& pair : src_info.handles) {
            driver_.map_memory(dst_info.dptr + current_offset, pair.second, pair.first);
            dst_info.handles.push_back(pair);
            current_offset += pair.second;
        }

        // 3. 不足分 (拡張差分)
        if (dst_info.mapped_size > src_info.mapped_size) {
            size_t diff = dst_info.mapped_size - src_info.mapped_size;
            
            // 差分もプールから探す
            CUmemGenericAllocationHandle h_new = pool_.pop(diff);
            if (h_new == 0) {
                h_new = driver_.create_physical_mem(diff);
            }
            
            driver_.map_memory(dst_info.dptr + current_offset, diff, h_new);
            dst_info.handles.push_back({h_new, diff});
        }

        // 4. src を空にする
        driver_.unmap_memory(src_info.dptr, src_info.mapped_size);
        src_info.handles.clear();
        src_info.mapped_size = 0;

        tracker_.update_alloc(dst, dst_info);
        tracker_.update_alloc((void*)src, src_info);

        LOG_INFO("REMAP", dst, count, "OPTIMIZED! Zero-Copy");
        return true;

    } catch (const std::exception& e) {
        LOG_ERR("REMAP_FAIL", dst, count, e.what());
        return false;
    }
}