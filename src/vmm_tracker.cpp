#include "../include/vmm_tracker.hpp"

void VmmTracker::register_alloc(void* ptr, AllocationInfo info) {
    std::lock_guard<std::mutex> lock(mtx_);
    map_[ptr] = info;
}
void VmmTracker::unregister_alloc(void* ptr) {
    std::lock_guard<std::mutex> lock(mtx_);
    map_.erase(ptr);
}
bool VmmTracker::get_alloc(void* ptr, AllocationInfo& out_info) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = map_.find(ptr);
    if (it == map_.end()) return false;
    out_info = it->second;
    return true;
}
void VmmTracker::update_alloc(void* ptr, const AllocationInfo& info) {
    std::lock_guard<std::mutex> lock(mtx_);
    map_[ptr] = info;
}