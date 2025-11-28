#pragma once
#include <string>

enum class VmmMode { VMM, MONITOR };
enum class LogLevel { ERROR = 0, INFO = 1, DEBUG = 2 };

class VmmEnv {
public:
    static VmmEnv& get();
    VmmMode get_mode() const { return mode_; }
    LogLevel get_log_level() const { return log_level_; }
    std::string get_log_file() const { return log_file_; }
    size_t get_initial_reserve_size() const { return reserve_size_; }
    
    // ★追加: 断片利用の許容比率 (0.0 ~ 1.0)
    double get_frag_ratio() const { return frag_ratio_; }

private:
    VmmEnv();
    VmmMode mode_;
    LogLevel log_level_;
    std::string log_file_;
    size_t reserve_size_;
    double frag_ratio_; // ★追加
};