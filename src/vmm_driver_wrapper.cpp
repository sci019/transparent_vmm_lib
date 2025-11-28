#include "../include/vmm_driver_wrapper.hpp"
#include "../include/vmm_logger.hpp" // ログ用
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

// ヘルパー: ポインタを文字列化
template<typename T>
std::string to_hex(T p) {
    std::stringstream ss;
    ss << "0x" << std::hex << (unsigned long long)p;
    return ss.str();
}

static void check_drv(CUresult res, const std::string& msg) {
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        throw std::runtime_error("[VMM] Driver API Error: " + msg + " : " + (errStr ? errStr : "Unknown"));
    }
}

void VmmDriverWrapper::init() {
    check_drv(cuInit(0), "cuInit");
}

size_t VmmDriverWrapper::get_granularity() {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    
    size_t gran;
    check_drv(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "GetGranularity");
    return gran;
}

size_t VmmDriverWrapper::get_total_memory() {
    CUdevice dev;
    check_drv(cuDeviceGet(&dev, 0), "DeviceGet");
    size_t bytes;
    check_drv(cuDeviceTotalMem(&bytes, dev), "DeviceTotalMem");
    return bytes;
}

CUdeviceptr VmmDriverWrapper::reserve_address(size_t size) {
    CUdeviceptr ptr;
    check_drv(cuMemAddressReserve(&ptr, size, 0, 0, 0), "AddressReserve");
    LOG_DBG("DRV_RES", (void*)ptr, size, "Reserved");
    return ptr;
}

void VmmDriverWrapper::free_address(CUdeviceptr ptr, size_t size) {
    check_drv(cuMemAddressFree(ptr, size), "AddressFree");
    LOG_DBG("DRV_FREE_VA", (void*)ptr, size, "Freed");
}

CUmemGenericAllocationHandle VmmDriverWrapper::create_physical_mem(size_t size) {
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    
    check_drv(cuMemCreate(&handle, size, &prop, 0), "MemCreate");
    LOG_DBG("DRV_CREATE_PA", nullptr, size, "Handle=" + to_hex(handle));
    return handle;
}

void VmmDriverWrapper::release_physical_mem(CUmemGenericAllocationHandle handle) {
    check_drv(cuMemRelease(handle), "MemRelease");
    LOG_DBG("DRV_REL_PA", nullptr, 0, "Handle=" + to_hex(handle));
}

void VmmDriverWrapper::map_memory(CUdeviceptr ptr, size_t size, CUmemGenericAllocationHandle handle) {
    // 1. Map
    CUresult res = cuMemMap(ptr, size, 0, handle, 0);
    if (res != CUDA_SUCCESS) {
        const char* errStr; cuGetErrorString(res, &errStr);
        std::string msg = "MemMap Failed. Ptr=" + to_hex(ptr) + " Size=" + std::to_string(size) + " Handle=" + to_hex(handle);
        throw std::runtime_error("[VMM] " + msg + " : " + (errStr ? errStr : "Unknown"));
    }
    
    // 2. SetAccess
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    res = cuMemSetAccess(ptr, size, &accessDesc, 1);
    if (res != CUDA_SUCCESS) {
        // 失敗時は即座にUnmapしてクリーンアップ
        cuMemUnmap(ptr, size);
        const char* errStr; cuGetErrorString(res, &errStr);
        std::string msg = "MemSetAccess Failed. Ptr=" + to_hex(ptr);
        throw std::runtime_error("[VMM] " + msg + " : " + (errStr ? errStr : "Unknown"));
    }

    LOG_DBG("DRV_MAP", (void*)ptr, size, "Handle=" + to_hex(handle));
}

void VmmDriverWrapper::unmap_memory(CUdeviceptr ptr, size_t size) {
    check_drv(cuMemUnmap(ptr, size), "MemUnmap");
    LOG_DBG("DRV_UNMAP", (void*)ptr, size, "");
}