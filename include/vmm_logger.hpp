#pragma once
#include "vmm_env.hpp"
#include <string>
#include <fstream>
#include <mutex>

class VmmLogger {
public:
    static VmmLogger& get();

    // ログ出力関数
    void log(LogLevel level, const std::string& action, void* ptr, size_t size, const std::string& extra = "");

private:
    VmmLogger() = default;
    ~VmmLogger();
    VmmLogger(const VmmLogger&) = delete;
    VmmLogger& operator=(const VmmLogger&) = delete;

    std::string levelToString(LogLevel l);

    std::ofstream ofs_;
    std::mutex mtx_;
};

// マクロ定義
#define LOG_ERR(action, ptr, size, msg) VmmLogger::get().log(LogLevel::ERROR, action, ptr, size, msg)
#define LOG_INFO(action, ptr, size, msg) VmmLogger::get().log(LogLevel::INFO, action, ptr, size, msg)
#define LOG_DBG(action, ptr, size, msg) VmmLogger::get().log(LogLevel::DEBUG, action, ptr, size, msg)