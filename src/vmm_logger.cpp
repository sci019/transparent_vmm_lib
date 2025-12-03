#include "../include/vmm_logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

VmmLogger& VmmLogger::get() {
    static VmmLogger instance;
    return instance;
}

VmmLogger::~VmmLogger() {
    if (ofs_.is_open()) ofs_.close();
}

void VmmLogger::log(LogLevel level, const std::string& action, void* ptr, size_t size, const std::string& extra) {
    // ログレベルフィルタ
    if (level > VmmEnv::get().get_log_level()) return;

    std::lock_guard<std::mutex> lock(mtx_);

    // ファイルオープン遅延処理
    if (!ofs_.is_open() && !VmmEnv::get().get_log_file().empty()) {
        ofs_.open(VmmEnv::get().get_log_file(), std::ios::out | std::ios::app);
    }

    std::ostream& out = ofs_.is_open() ? ofs_ : std::cerr;
    
    // 時刻取得
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // フォーマット出力: [TIME] [LEVEL] [ACTION] Ptr=... Size=... Extra
    out << "[" << std::put_time(std::localtime(&t_c), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
        << "[" << levelToString(level) << "] "
        << "[" << action << "] "
        << "Ptr=" << ptr << " "
        << "Size=" << size;
    
    if (!extra.empty()) out << " " << extra;
    out << std::endl;
}

std::string VmmLogger::levelToString(LogLevel l) {
    switch(l) {
        case LogLevel::ERROR: return "ERR";
        case LogLevel::INFO:  return "INF";
        case LogLevel::DEBUG: return "DBG";
        default: return "UNK";
    }
}