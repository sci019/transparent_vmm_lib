#include "../include/vmm_pool.hpp"
#include "../include/vmm_driver_wrapper.hpp"
#include "../include/vmm_logger.hpp" // ログ用
#include <string>
#include <sstream>

template<typename T>
std::string to_hex(T p) {
    std::stringstream ss; ss << "0x" << std::hex << (unsigned long long)p; return ss.str();
}

CUmemGenericAllocationHandle VmmPhysicalPool::pop(size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.find(size);
    if (it != pool_.end()) {
        CUmemGenericAllocationHandle h = it->second;
        pool_.erase(it);
        LOG_DBG("POOL_POP", nullptr, size, "Hit Handle=" + to_hex(h));
        return h;
    }
    return 0;
}

bool VmmPhysicalPool::pop_largest_le(size_t req_size, CUmemGenericAllocationHandle* out_handle, size_t* out_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pool_.upper_bound(req_size);
    if (it == pool_.begin()) {
        return false;
    }
    --it;
    *out_size = it->first;
    *out_handle = it->second;
    pool_.erase(it);
    LOG_DBG("POOL_POP_FRAG", nullptr, *out_size, "Req=" + std::to_string(req_size) + " Handle=" + to_hex(*out_handle));
    return true;
}

void VmmPhysicalPool::push(size_t size, CUmemGenericAllocationHandle handle) {
    std::lock_guard<std::mutex> lock(mtx_);
    pool_.insert({size, handle});
    LOG_DBG("POOL_PUSH", nullptr, size, "Handle=" + to_hex(handle));
}

size_t VmmPhysicalPool::drain(VmmDriverWrapper& driver) {
    std::lock_guard<std::mutex> lock(mtx_);
    size_t count = 0;
    size_t total_bytes = 0;

    LOG_DBG("POOL_DRAIN_START", nullptr, 0, "Count=" + std::to_string(pool_.size()));

    for (auto& pair : pool_) {
        driver.release_physical_mem(pair.second);
        total_bytes += pair.first;
        count++;
    }
    pool_.clear();
    
    LOG_DBG("POOL_DRAIN_END", nullptr, total_bytes, "Released Count=" + std::to_string(count));
    return total_bytes;
}