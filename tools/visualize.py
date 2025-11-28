import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import argparse
from datetime import datetime

def parse_log(filepath):
    events = []
    pattern = re.compile(r'\[(.*?)\] \[(.*?)\] \[(.*?)\] Ptr=(0x[0-9a-fA-F]+|[0-9]+) Size=(\d+)(.*)')
    start_time = None
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m: continue
            time_str, level, action, ptr_str, size_str, extra = m.groups()
            ptr = int(ptr_str, 16) if ptr_str.startswith('0x') else int(ptr_str)
            size = int(size_str)
            dt = datetime.strptime(time_str, "%H:%M:%S.%f")
            if start_time is None: start_time = dt
            events.append({'time': (dt - start_time).total_seconds(), 'action': action, 'ptr': ptr, 'size': size})
    return events

def plot_timeline(events, output):
    fig, ax = plt.subplots(figsize=(14, 6))
    active = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    c_idx = 0; max_time = 0.1; min_a = float('inf'); max_a = 0

    for ev in events:
        ptr, act, t, sz = ev['ptr'], ev['action'], ev['time'], ev['size']
        max_time = max(max_time, t)
        if act == "ALLOC":
            active[ptr] = {'s': t, 'sz': sz, 'c': colors[c_idx%5]}
            c_idx+=1
            min_a = min(min_a, ptr); max_a = max(max_a, ptr+sz)
        elif act == "FREE" and ptr in active:
            info = active[ptr]
            ax.add_patch(patches.Rectangle((info['s'], ptr), t-info['s'], info['sz'], facecolor=info['c'], edgecolor='black', alpha=0.6))
            del active[ptr]
        elif act == "REMAP":
            active[ptr] = {'s': t, 'sz': sz, 'c': '#ff0000'}
            ax.text(t, ptr+sz, "REMAP", color='red', fontsize=8)
            min_a = min(min_a, ptr); max_a = max(max_a, ptr+sz)

    for ptr, info in active.items():
        ax.add_patch(patches.Rectangle((info['s'], ptr), max_time-info['s']+0.1, info['sz'], facecolor=info['c'], edgecolor='black', alpha=0.6))

    ax.set_xlabel('Time(s)'); ax.set_ylabel('Virtual Address')
    if min_a != float('inf'): plt.ylim(min_a, max_a*1.1)
    plt.xlim(0, max_time*1.1); plt.grid(True, alpha=0.3)
    plt.savefig(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log"); parser.add_argument("out")
    args = parser.parse_args()
    plot_timeline(parse_log(args.log), args.out)