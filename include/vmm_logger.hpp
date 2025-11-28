#pragma once
#include "vmm_env.hpp"
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>

class VmmLogger {
public:
    static VmmLogger& get() {
        static VmmLogger instance;
        return instance;
    }

    template <typename... Args>
    void log(LogLevel level, const std::string& action, void* ptr, size_t size, const std::string& extra = "") {
        if (level > VmmEnv::get().get_log_level()) return;

        std::lock_guard<std::mutex> lock(mtx_);
        if (!ofs_.is_open() && !VmmEnv::get().get_log_file().empty()) {
            ofs_.open(VmmEnv::get().get_log_file(), std::ios::out | std::ios::app);
        }

        std::ostream& out = ofs_.is_open() ? ofs_ : std::cerr;
        auto now = std::chrono::system_clock::now();
        auto t_c = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        out << "[" << std::put_time(std::localtime(&t_c), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
            << "[" << levelToString(level) << "] "
            << "[" << action << "] "
            << "Ptr=" << ptr << " "
            << "Size=" << size;
        if (!extra.empty()) out << " " << extra;
        out << std::endl;
    }

private:
    VmmLogger() = default;
    ~VmmLogger() { if (ofs_.is_open()) ofs_.close(); }

    std::string levelToString(LogLevel l) {
        switch(l) {
            case LogLevel::ERROR: return "ERR";
            case LogLevel::INFO:  return "INF";
            case LogLevel::DEBUG: return "DBG";
            default: return "UNK";
        }
    }
    std::ofstream ofs_;
    std::mutex mtx_;
};

#define LOG_ERR(action, ptr, size, msg) VmmLogger::get().log(LogLevel::ERROR, action, ptr, size, msg)
#define LOG_INFO(action, ptr, size, msg) VmmLogger::get().log(LogLevel::INFO, action, ptr, size, msg)
#define LOG_DBG(action, ptr, size, msg) VmmLogger::get().log(LogLevel::DEBUG, action, ptr, size, msg)