#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define MB (1024UL * 1024UL)

int main() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Initial Free Memory: " << free_mem / MB << " MB" << std::endl;

    size_t chunk_size = 64 * MB;
    size_t target_fill = (size_t)(total_mem * 0.85);
    int num_chunks = target_fill / chunk_size;
    
    std::cout << "Allocating " << num_chunks << " chunks of " << chunk_size / MB << " MB..." << std::endl;
    std::vector<void*> ptrs;
    ptrs.reserve(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        void* p;
        if (cudaMalloc(&p, chunk_size) != cudaSuccess) {
            std::cout << "\nWarning: Saturation at " << i << std::endl;
            break;
        }
        ptrs.push_back(p);
        if (i % (num_chunks/20 + 1) == 0) std::cout << "." << std::flush;
    }
    std::cout << " Done." << std::endl;

    std::cout << "Creating fragmentation (freeing every 2nd chunk)..." << std::endl;
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        if (ptrs[i]) {
            cudaFree(ptrs[i]);
            ptrs[i] = nullptr;
        }
    }

    size_t huge_alloc_size = (size_t)((ptrs.size()/2 * chunk_size) * 0.9); 
    std::cout << "Attempting huge allocation: " << huge_alloc_size / MB << " MB..." << std::endl;

    void* huge_ptr;
    if (cudaMalloc(&huge_ptr, huge_alloc_size) == cudaSuccess) {
        std::cout << "SUCCESS: Huge allocation succeeded!" << std::endl;
        cudaFree(huge_ptr);
    } else {
        std::cout << "FAILURE: Huge allocation failed!" << std::endl;
    }

    for (void* p : ptrs) if (p) cudaFree(p);
    return 0;
}