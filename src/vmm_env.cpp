#include "../include/vmm_env.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

VmmEnv& VmmEnv::get() {
    static VmmEnv instance;
    return instance;
}

VmmEnv::VmmEnv() {
    // Mode
    const char* mode = std::getenv("VMM_MODE");
    mode_ = (mode && std::string(mode) == "MONITOR") ? VmmMode::MONITOR : VmmMode::VMM;

    // Log Level
    const char* lvl = std::getenv("VMM_LOG_LEVEL");
    log_level_ = LogLevel::ERROR;
    if (lvl) {
        std::string s(lvl);
        if (s == "INFO") log_level_ = LogLevel::INFO;
        else if (s == "DEBUG") log_level_ = LogLevel::DEBUG;
    }

    // Log File
    const char* file = std::getenv("VMM_LOG_FILE");
    log_file_ = file ? std::string(file) : "";

    // Reserve Size (Default: 4GB)
    const char* rsv = std::getenv("VMM_RESERVE_SIZE_MB");
    reserve_size_ = rsv ? std::stoull(rsv) * 1024 * 1024 : 4ULL * 1024 * 1024 * 1024;

    // ★追加: Fragment Ratio (Default: 0.25 = 25%)
    // 要求サイズの 25% 未満のゴミは使わずに新規作成する
    const char* ratio = std::getenv("VMM_FRAG_RATIO");
    frag_ratio_ = ratio ? std::stod(ratio) : 0.25;
}