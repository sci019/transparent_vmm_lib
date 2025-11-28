# CUDA VMM Transparent Hook Library

## 概要
本プロジェクトは、`LD_PRELOAD` を利用して標準の `cudaMalloc` をフックし、CUDA Driver API の **Virtual Memory Management (VMM)** 機能（`cuMemCreate`, `cuMemMap` 等）へ置き換えるライブラリである。

vLLM などの最新推論エンジンで採用されているメモリ管理手法を、**既存のアプリケーションバイナリに一切手を加えることなく（変更ゼロで）** 適用することを目的とする。ソースコードが存在しない、あるいは改変が許されない極秘アプリケーションであっても、本ライブラリをロードするだけで動作する。

本バージョンでは **完全透過的 (Transparent)** な実装を行っており、アプリケーションは従来の `cudaMalloc` / `cudaMemcpy` を使用するだけで、裏側で自動的に VMM による最適化が適用される。

## 主な機能と目的

### 1. フラグメンテーション（断片化）による OOM の回避
Runtime API (`cudaMalloc`) は通常、物理的に連続した領域を要求する。対して本ライブラリは、Driver API を用いて以下の手順で確保を行う。

1.  **仮想アドレス予約 (`cuMemAddressReserve`)**: 巨大な仮想アドレス空間のみを先に確保する。
2.  **物理メモリ割当 (`cuMemCreate`)**: 必要な分だけ物理メモリハンドルを作成する（物理的に不連続で良い）。
3.  **マッピング (`cuMemMap`)**: 不連続な物理メモリを、連続した仮想アドレスに紐付ける。

これにより、VRAM が断片化していても、合計容量さえ空いていれば巨大な確保が可能となり、OOM を回避できる。

### 2. ゼロコピー・リサイズによる高速化
従来のリサイズ処理（`Malloc` -> `Memcpy` -> `Free`）はデータ移動のコストが高い。
本ライブラリは、初期確保時に将来の拡張を見越して仮想アドレス（VA）を大きく予約している。
アプリケーションが「リサイズ（再確保＋全データコピー）」を行おうとした瞬間、ライブラリがそれを検知し、**「物理メモリの紐付け（マッピング）を付け替える」** 処理を裏で行う。
これにより、実際のデータコピーが発生しないため、拡張コストはほぼゼロになる。

## アーキテクチャと機能分離

本ライブラリは保守性と拡張性を考慮し、以下のモジュールに分離して実装されている。

* **VmmDriverWrapper**: 複雑なドライバAPI (`cuMem*` 系) の操作を隠蔽する低レイヤラッパー。
* **VmmTracker**: 「どの仮想アドレスがどの物理ハンドルを持っているか」という状態管理を行う。
* **VmmOptimizer**: `cudaMemcpy` 呼び出し時に介入し、ゼロコピー化が可能かどうかの判定と実行（Remap）を行うロジックの中枢。
* **VmmManager**: 上記コンポーネントを統括するファサード。
* **Hook**: `cudaMalloc`, `cudaFree`, `cudaMemcpy` をフックし、Manager へ処理を委譲するエントリポイント。

## ディレクトリ構成

```text
.
├── include/
│   ├── vmm_env.hpp             # 環境変数管理クラス定義
│   ├── vmm_logger.hpp          # ロガークラス定義
│   ├── vmm_driver_wrapper.hpp  # ドライバAPIラッパー定義
│   ├── vmm_tracker.hpp         # 状態管理クラス定義
│   ├── vmm_optimizer.hpp       # 最適化ロジック定義
│   └── vmm_manager.hpp         # 統合管理クラス定義
├── src/
│   ├── vmm_env.cpp
│   ├── vmm_driver_wrapper.cpp
│   ├── vmm_tracker.cpp
│   ├── vmm_optimizer.cpp
│   ├── vmm_manager.cpp
│   └── hook.cpp                # フックエントリポイント
├── test/
│   ├── frag_test.cu            # OOM回避の実証テスト
│   └── resize_test.cu          # 自動ゼロコピー性能テスト
├── tools/
│   └── visualize.py            # ログ可視化・グラフ生成ツール
├── lib/                        # ビルド生成物 (.so)
├── bin/                        # ビルド生成物 (実行ファイル)
└── *.sh                        # 自動化スクリプト
```

## ビルドと実行

### ビルド
```bash
chmod +x build.sh run_tests.sh
./build.sh
```

### テスト実行
```bash
./run_tests.sh
```

スクリプトは以下の2つのテストを自動実行する。

1.  **フラグメンテーション耐性テスト (`frag_test`)**
    * メモリを虫食い状態にし、標準 `cudaMalloc` では失敗する巨大確保が、本ライブラリ経由（VMM）では成功することを実証する。
2.  **リサイズ性能テスト (`resize_test`)**
    * 「確保・コピー・解放」の古典的手順を実行し、本ライブラリがそれを自動検知して高速化した結果（処理時間）を表示する。

## 技術的詳細と自動最適化ロジック

### フックされるAPI
* `cudaMalloc`: VMM マネージャー経由でメモリを確保する。
* `cudaFree`: VMM マネージャー経由でメモリを解放する。
* `cudaMemcpy`: データ転送を監視し、最適化可能なパターンを検知する。

### 自動最適化の発動条件
アプリケーションがドライバAPIや本ライブラリを意識する必要はない。以下のパターンが検出された場合、自動的に最適化が発動する。

1.  `ptrA = cudaMalloc(SizeA)` : データが存在
2.  `ptrB = cudaMalloc(SizeB)` : 新規確保 (SizeB >= SizeA)
3.  `cudaMemcpy(ptrB, ptrA, SizeA, cudaMemcpyDeviceToDevice)` : **ここをフック**

通常であればデータをVRAM内でコピーするが、本ライブラリは `ptrA` に紐づく物理メモリを `ptrB` にマップし直す（Remap）。
その後、`ptrA` は「空の殻（物理実体のない仮想アドレス）」となり、後の `cudaFree(ptrA)` で安全に破棄される。

## 実行モードと環境変数

本ライブラリは以下の環境変数によって挙動を制御できる。

### `VMM_MODE`
* **`VMM`** (Default):
    VMM 機能（OOM回避・ゼロコピーリサイズ）を有効にする。通常利用時はこちらを使用する。
* **`MONITOR`**:
    標準の CUDA API (`cudaMalloc` 等) をそのまま実行し、メモリ確保のログのみを出力する。最適化前後の挙動比較（ベースライン作成）に使用する。

### `VMM_LOG_LEVEL`
* **`ERROR`**: エラー発生時のみ出力。
* **`INFO`**: 可視化ツールでのグラフ生成に必要な最小限のログを出力。
* **`DEBUG`**: 内部動作追跡用の詳細ログを出力。

### `VMM_LOG_FILE`
* ログの出力先ファイルパスを指定する。未指定の場合は標準エラー出力 (`stderr`) に出力される。

## 可視化ツール (Visualization)

`tools/visualize.py` を使用して、生成されたログファイルから「仮想アドレス空間の時系列変化」を可視化したグラフ（PNG画像）を生成できる。

付属の `run_tests.sh` は、`MONITOR` モード（最適化なし）と `VMM` モード（最適化あり）の両方を自動実行し、以下の比較画像を生成する。

* **graph_frag_monitor.png**: 通常の `cudaMalloc` 時のメモリ配置。断片化（虫食い）の様子が確認できる。
* **graph_frag_vmm.png**: VMM 適用時のメモリ配置。物理メモリが不連続でも、仮想アドレス上では効率的に配置されている様子が確認できる。

## 要件
* Linux (x86_64)
* NVIDIA GPU (Pascal アーキテクチャ以降推奨)
* CUDA Toolkit 11.0 以上