#!/bin/bash
mkdir -p lib bin
CUDA_PATH=/usr/local/cuda

SRCS="src/vmm_env.cpp src/vmm_logger.cpp src/vmm_driver_wrapper.cpp src/vmm_allocator.cpp src/vmm_pool.cpp src/vmm_tracker.cpp src/vmm_optimizer.cpp src/vmm_manager.cpp src/hook.cpp"

echo "Building VMM Hook Library..."
g++ -shared -fPIC -o lib/libvmm_hook.so $SRCS \
    -I./include -I$CUDA_PATH/include -L$CUDA_PATH/lib64 -lcuda -lcudart -ldl

echo "Building Tests..."
nvcc -o bin/frag_test test/frag_test.cu
nvcc -o bin/resize_test test/resize_test.cu