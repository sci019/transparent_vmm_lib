#!/bin/bash
set -e
./build.sh
LIB=$(pwd)/lib/libvmm_hook.so

# ★詳細ログのために DEBUG レベルに変更
export VMM_LOG_LEVEL=DEBUG
export VMM_RESERVE_SIZE_MB=16384

echo "#########################################################"
echo "# 1. Resize Performance Test (1.5GB -> 3GB) Loop x10"
echo "#########################################################"

echo "[Resize] Running MONITOR Mode..."
export VMM_MODE=MONITOR
export VMM_LOG_FILE=resize_monitor.log
rm -f $VMM_LOG_FILE
LD_PRELOAD=$LIB bin/resize_test

echo "[Resize] Running VMM Mode..."
export VMM_MODE=VMM
export VMM_LOG_FILE=resize_vmm.log
rm -f $VMM_LOG_FILE
LD_PRELOAD=$LIB bin/resize_test

echo ""
echo "#########################################################"
echo "# 2. Fragmentation Test"
echo "#########################################################"
# Monitor
export VMM_MODE=MONITOR
export VMM_LOG_FILE=frag_monitor.log
rm -f $VMM_LOG_FILE
set +e
LD_PRELOAD=$LIB bin/frag_test
set -e
# VMM
export VMM_MODE=VMM
export VMM_LOG_FILE=frag_vmm.log
rm -f $VMM_LOG_FILE
LD_PRELOAD=$LIB bin/frag_test

echo ""
echo "#########################################################"
echo "# 3. Graphs"
echo "#########################################################"
# DEBUGログになってもvisualize.pyは必要な行だけパースするので動作する
python3 tools/visualize.py resize_monitor.log graph_resize_monitor.png
python3 tools/visualize.py resize_vmm.log graph_resize_vmm.png
python3 tools/visualize.py frag_monitor.log graph_frag_monitor.png
python3 tools/visualize.py frag_vmm.log graph_frag_vmm.png

echo "Done."