#include "../include/vmm_manager.hpp"
#include "../include/vmm_logger.hpp"
#include "../include/vmm_env.hpp"
#include <mutex>

VmmManager& VmmManager::get() {
    static VmmManager instance;
    return instance;
}

VmmManager::VmmManager() 
    : optimizer_(driver_, tracker_, allocator_, pool_) 
{
}

void VmmManager::init() {
    std::call_once(init_flag_, [&](){ 
        driver_.init(); 
        granularity_ = driver_.get_granularity(); 
        
        pool_size_ = VmmEnv::get().get_initial_reserve_size();
        pool_base_ = driver_.reserve_address(pool_size_);
        allocator_.init(pool_base_, pool_size_);

        LOG_INFO("INIT", (void*)pool_base_, pool_size_, "VA Pool Reserved");
    });
}

cudaError_t VmmManager::allocate(void** dptr, size_t size) {
    try {
        init();
        size_t aligned_size = ((size + granularity_ - 1) / granularity_) * granularity_;

        bool retried = false;
        AllocationInfo info;
        info.va_size = aligned_size;
        info.mapped_size = aligned_size;
        info.user_size = size;

        // ★VA再取得戦略: ループ内で毎回新しいVAを取得する
        while (true) {
            // 1. VA確保
            CUdeviceptr ptr = allocator_.alloc(aligned_size);
            if (ptr == 0) {
                // VAプール自体が枯渇した場合はどうしようもない (DrainしてもVAは増えない)
                LOG_ERR("ALLOC_OOM", nullptr, size, "VA Pool Exhausted");
                return cudaErrorMemoryAllocation;
            }
            info.dptr = ptr; // 確保したポインタをセット

            info.handles.clear();
            size_t current_offset = 0;
            
            try {
                LOG_DBG("ALLOC_START", (void*)ptr, size, "Retry=" + std::to_string(retried));

                // --- PA確保 & Map (Scatter-Gather) ---
                size_t remaining = aligned_size;
                double frag_ratio = VmmEnv::get().get_frag_ratio();
                
                while (remaining > 0) {
                    CUmemGenericAllocationHandle handle = 0;
                    size_t chunk_size = 0;

                    // A. 完全一致
                    handle = pool_.pop(remaining);
                    if (handle != 0) {
                        chunk_size = remaining;
                    } 
                    else {
                        // B. 断片利用
                        if (pool_.pop_largest_le(remaining, &handle, &chunk_size)) {
                            if (chunk_size < (size_t)(remaining * frag_ratio)) {
                                pool_.push(chunk_size, handle);
                                handle = 0;
                            }
                        }
                        // C. 新規作成
                        if (handle == 0) {
                            chunk_size = remaining;
                            handle = driver_.create_physical_mem(chunk_size);
                        }
                    }

                    // マップ実行
                    driver_.map_memory(ptr + current_offset, chunk_size, handle);
                    
                    info.handles.push_back({handle, chunk_size});
                    current_offset += chunk_size;
                    remaining -= chunk_size;
                }
                
                // 成功したらループを抜ける
                break;

            } catch (const std::exception& e) {
                // --- エラーハンドリング ---
                LOG_DBG("ALLOC_ERR", (void*)ptr, current_offset, std::string("Error: ") + e.what());

                // 1. 途中までマップした分を解除
                if (current_offset > 0) {
                    driver_.unmap_memory(ptr, current_offset);
                }
                
                // 2. 今回確保したハンドルを破棄 (プールには戻さない)
                for (auto& pair : info.handles) {
                    driver_.release_physical_mem(pair.first);
                }
                info.handles.clear();

                // 3. ★重要: 汚れたVAを返却 (捨てる)
                // 次のループでは新しいVAが割り当てられるので、重複マップなどの不整合を防げる
                allocator_.free(ptr, aligned_size);
                LOG_DBG("VA_DISCARD", (void*)ptr, aligned_size, "Discarded dirty VA");

                // 4. リトライ判定
                if (!retried) {
                    LOG_INFO("POOL_DRAIN", nullptr, 0, "OOM detected. Draining pool...");
                    size_t drained_bytes = pool_.drain(driver_);
                    LOG_INFO("POOL_DRAIN", nullptr, drained_bytes, "Drained bytes. Retrying...");
                    
                    retried = true;
                    continue; // ループ先頭へ (VA再確保からやり直し)
                } else {
                    throw; // 2回目もダメなら諦める
                }
            }
        }

        tracker_.register_alloc((void*)info.dptr, info);
        *dptr = (void*)info.dptr;

        LOG_INFO("ALLOC", (void*)info.dptr, size, "PA=" + std::to_string(aligned_size) + " Frags=" + std::to_string(info.handles.size()));
        return cudaSuccess;

    } catch (const std::exception& e) {
        LOG_ERR("ALLOC_FAIL", nullptr, size, e.what());
        return cudaErrorMemoryAllocation;
    }
}

cudaError_t VmmManager::free(void* dptr) {
    try {
        AllocationInfo info;
        if (!tracker_.get_alloc(dptr, info)) return cudaSuccess;

        LOG_INFO("FREE", dptr, info.mapped_size, "VMM");

        if (info.mapped_size > 0) {
            driver_.unmap_memory(info.dptr, info.mapped_size);
        }
        
        for (auto& pair : info.handles) {
            pool_.push(pair.second, pair.first);
        }
        
        allocator_.free(info.dptr, info.va_size);
        tracker_.unregister_alloc(dptr);
        return cudaSuccess;
    } catch (const std::exception& e) {
        LOG_ERR("FREE_FAIL", dptr, 0, e.what());
        return cudaErrorUnknown;
    }
}

bool VmmManager::try_optimize_memcpy(void* dst, const void* src, size_t count) {
    return optimizer_.try_optimize_copy(dst, src, count);
}