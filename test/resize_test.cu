#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

// テスト設定: 1.5GB -> 3GB (一時的な合計使用量 4.5GB)
// VRAM 6GB 環境において、OS領域を除いて確保可能な最大規模のサイズ設定
#define MB (1024UL * 1024UL)
#define START_SIZE (1536 * MB) // 1.5GB
#define END_SIZE   (3072 * MB) // 3.0GB
#define LOOP_COUNT 10

// 高分解能タイマー
auto now() { return std::chrono::high_resolution_clock::now(); }

// C++11/14互換のためのテンプレート関数 (ミリ秒計測)
template <typename T>
double diff_ms(T start, T end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== Resize Test (1.5GB -> 3GB) x " << LOOP_COUNT << " loops ===" << std::endl;
    
    std::vector<double> times;
    
    for (int i = 0; i < LOOP_COUNT; ++i) {
        void *d_ptr_old, *d_ptr_new;

        // 1. 初期領域の確保 (1.5GB)
        // 2回目以降はプールから高速に確保される
        if (cudaMalloc(&d_ptr_old, START_SIZE) != cudaSuccess) {
            std::cerr << "Loop " << i << ": Failed to malloc start size" << std::endl; return 1;
        }
        
        // 初回のみデータを書き込み、物理メモリを確実に割り当てさせる
        // (2回目以降はプール再利用のため、データは残っているが上書きとみなして省略可)
        if (i == 0) {
            cudaMemset(d_ptr_old, 0xAA, START_SIZE);
            cudaDeviceSynchronize();
        }

        auto t1 = now();

        // 2. 拡張領域の確保 (3GB)
        // VMMモードではプール内の断片化メモリを結合して確保するため、高速化が期待される
        if (cudaMalloc(&d_ptr_new, END_SIZE) != cudaSuccess) {
            std::cerr << "Loop " << i << ": Failed to malloc end size" << std::endl; 
            cudaFree(d_ptr_old);
            return 1;
        }

        // 3. データコピー (最適化の対象)
        // VMMモード: 物理メモリのリマップ (Remap) を行うため、データ移動は発生しない
        // Monitorモード: 実際にデータをコピーするため、帯域幅に応じた時間がかかる
        cudaMemcpy(d_ptr_new, d_ptr_old, START_SIZE, cudaMemcpyDeviceToDevice);

        // 4. 旧領域の解放
        // VMMモードでは中身（物理メモリ）は移動済みのため、仮想アドレスのみが解放される
        cudaFree(d_ptr_old);

        cudaDeviceSynchronize();
        auto t2 = now();

        double ms = diff_ms(t1, t2);
        times.push_back(ms);
        std::cout << "Loop " << i+1 << ": " << ms << " ms" << std::endl;

        // 5. 拡張領域の解放
        // ここで3GB分の物理メモリがプールに返却され、次回のループで再利用される
        cudaFree(d_ptr_new);
    }

    // 初回はプールが空のため遅い可能性があるが、平均をとることでプールの効果を確認する
    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Average Time: " << avg << " ms" << std::endl;
    return 0;
}