#include <array>
#include <cctype>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;

struct Config {
    std::string suite = "final";
    std::string baseline = "qtn_baseline.py";
    std::string python = "python";
    std::string log_dir = "logs_cpp";
    int nqubits = 30;
    int ngates = 10000;
    int max_bond = 64;
    std::string svd_cutoff = "1e-8";
    int threads = 8;
    std::string optimizer = "auto-hq";
    std::string expectation_engine = "local";
    std::string two_qubit_apply = "name";
    int repeats = 3;
    int timeout_sec = 300;
    bool dry_run = false;
    bool profile_conversion = false;
    bool fuse_single_qubit = false;
    bool absorb_1q_into_2q = false;
    std::vector<std::string> observables = {"z0", "zz-chain"};
    std::vector<int> bonds = {8, 16, 32, 64, 128, 256};
    std::vector<std::string> cutoffs = {"1e-6", "1e-8", "1e-10"};
    std::vector<int> thread_list = {1, 2, 4, 8, 16, 32};
    std::vector<std::string> optimizers = {"greedy", "auto", "auto-hq", "optimal"};
    bool batch = false;
};

struct Case {
    std::string mode = "mps";
    std::string output = "expectation";
    std::string workload = "random-local";
    int nqubits = 30;
    int ngates = 10000;
    std::string observable = "z0";
    int max_bond = 64;
    bool has_max_bond = true;
    std::string svd_cutoff = "1e-8";
    int threads = 8;
    std::string optimizer = "auto-hq";
    std::string expectation_engine = "local";
    std::string two_qubit_apply = "name";
    bool check_ref = false;
    bool profile_conversion = false;
    bool fuse_single_qubit = false;
    bool absorb_1q_into_2q = false;
    int check_ref_max_qubits = 20;
    int timeout_sec = 0;
};

static std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out += ch;
        }
    }
    out += "'";
    return out;
}

static std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool file_exists_nonempty(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::file_size(path, ec) > 0;
}

static std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return ss.str();
}

static std::string sanitize(std::string value) {
    for (char& ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_')) {
            ch = '_';
        }
    }
    return value;
}

static std::vector<std::string> split_strings(std::string text) {
    for (char& ch : text) {
        if (ch == ',') {
            ch = ' ';
        }
    }
    std::istringstream iss(text);
    std::vector<std::string> out;
    std::string item;
    while (iss >> item) {
        out.push_back(item);
    }
    return out;
}

static std::vector<int> split_ints(const std::string& text) {
    std::vector<int> out;
    for (const auto& item : split_strings(text)) {
        out.push_back(std::stoi(item));
    }
    return out;
}

static std::string csv_escape(const std::string& value) {
    bool quote = false;
    for (char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            quote = true;
            break;
        }
    }
    if (!quote) {
        return value;
    }
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

static std::string json_escape(const std::string& value) {
    std::string out;
    for (char ch : value) {
        if (ch == '\\') out += "\\\\";
        else if (ch == '"') out += "\\\"";
        else if (ch == '\n') out += "\\n";
        else if (ch == '\r') out += "\\r";
        else if (ch == '\t') out += "\\t";
        else out += ch;
    }
    return out;
}

static std::string regex_first(const std::string& text, const std::regex& pattern) {
    std::smatch match;
    if (std::regex_search(text, match, pattern) && match.size() > 1) {
        return match[1].str();
    }
    return "";
}

static std::string json_string(const std::string& text, const std::string& key) {
    return regex_first(text, std::regex("\"" + key + "\"\\s*:\\s*\"([^\"]*)\""));
}

static std::string json_number(const std::string& text, const std::string& key) {
    return regex_first(
        text,
        std::regex("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)")
    );
}

static std::string json_bool(const std::string& text, const std::string& key) {
    return regex_first(text, std::regex("\"" + key + "\"\\s*:\\s*(true|false)"));
}

static std::string json_number_after(
    const std::string& text,
    const std::string& marker,
    const std::string& key
) {
    const auto pos = text.find(marker);
    if (pos == std::string::npos) {
        return "";
    }
    return json_number(text.substr(pos), key);
}

static std::string json_bool_after(
    const std::string& text,
    const std::string& marker,
    const std::string& key
) {
    const auto pos = text.find(marker);
    if (pos == std::string::npos) {
        return "";
    }
    return json_bool(text.substr(pos), key);
}

static std::string time_field(const std::string& text, const std::string& label) {
    return regex_first(text, std::regex(label + ".*:\\s*([^\\n\\r]+)"));
}

static std::string map_get(
    const std::map<std::string, std::string>& row,
    const std::string& key
) {
    const auto it = row.find(key);
    return it == row.end() ? "" : it->second;
}

static int system_return_code(int raw) {
    if (raw == -1) {
        return -1;
    }
#ifdef _WIN32
    return raw;
#else
    if (WIFEXITED(raw)) {
        return WEXITSTATUS(raw);
    }
    if (WIFSIGNALED(raw)) {
        return 128 + WTERMSIG(raw);
    }
    return raw;
#endif
}

static std::vector<std::string> csv_fields() {
    return {
        "timestamp",
        "mode",
        "output",
        "workload",
        "nqubits",
        "requested_ngates",
        "max_bond",
        "svd_cutoff",
        "observable",
        "threads",
        "optimizer",
        "expectation_engine",
        "two_qubit_apply",
        "expectation_terms",
        "expectation_engine_fallback",
        "returncode",
        "batch_run",
        "used_usr_bin_time",
        "stdout_log",
        "stderr_log",
        "elapsed_sec_internal",
        "stage_total_sec",
        "stage_set_threads_sec",
        "stage_set_backend_sec",
        "stage_build_circuit_sec",
        "stage_execute_sec",
        "stage_qibo_to_quimb_sec",
        "stage_quimb_circuit_init_sec",
        "stage_gate_loop_sec",
        "stage_gate_metadata_sec",
        "stage_gate_apply_sec",
        "stage_gate_apply_1q_sec",
        "stage_gate_apply_2q_sec",
        "stage_gate_apply_other_sec",
        "stage_gate_fusion_matrix_sec",
        "stage_gate_absorb_matrix_sec",
        "stage_expectation_contract_sec",
        "stage_mps_batched_attempt_sec",
        "actual_gate_count",
        "gate_count_1q",
        "gate_count_2q",
        "gate_count_other",
        "gate_count_measure",
        "gate_count_applied_1q",
        "gate_count_applied_2q",
        "gate_count_applied_other",
        "gate_count_absorbed_1q",
        "expectation_value",
        "state_norm",
        "qibo",
        "qibotn",
        "qibojit",
        "quimb",
        "cotengra",
        "check_ref",
        "ref_elapsed_sec",
        "max_abs_error",
        "allclose_1e-8",
        "expected_expectation_value",
        "abs_error",
        "isclose_1e-8",
        "time_elapsed_wall",
        "time_user_sec",
        "time_system_sec",
        "cpu_percent",
        "max_rss_kb",
        "batch_time_elapsed_wall",
        "batch_max_rss_kb",
        "minor_page_faults",
        "major_page_faults",
        "status",
    };
}

static void append_csv(const fs::path& path, const std::map<std::string, std::string>& row) {
    const bool write_header = !file_exists_nonempty(path);
    std::ofstream out(path, std::ios::app);
    const auto fields = csv_fields();
    if (write_header) {
        for (size_t i = 0; i < fields.size(); ++i) {
            if (i) out << ',';
            out << fields[i];
        }
        out << '\n';
    }
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i) out << ',';
        auto it = row.find(fields[i]);
        if (it != row.end()) {
            out << csv_escape(it->second);
        }
    }
    out << '\n';
}

static void append_jsonl(const fs::path& path, const std::map<std::string, std::string>& row) {
    std::ofstream out(path, std::ios::app);
    out << "{";
    bool first = true;
    for (const auto& [key, value] : row) {
        if (!first) out << ",";
        first = false;
        out << "\"" << json_escape(key) << "\":\"" << json_escape(value) << "\"";
    }
    out << "}\n";
}

static std::string case_json(const Case& c, int index) {
    std::ostringstream out;
    out << "{";
    out << "\"case_name\":\"" << json_escape(std::to_string(index) + "_" + c.observable) << "\"";
    out << ",\"mode\":\"" << json_escape(c.mode) << "\"";
    out << ",\"output\":\"" << json_escape(c.output) << "\"";
    out << ",\"workload\":\"" << json_escape(c.workload) << "\"";
    out << ",\"nqubits\":" << c.nqubits;
    if (c.ngates >= 0) {
        out << ",\"ngates\":" << c.ngates;
    }
    out << ",\"observable\":\"" << json_escape(c.observable) << "\"";
    if (c.has_max_bond) {
        out << ",\"max_bond\":" << c.max_bond;
    } else {
        out << ",\"max_bond\":null";
    }
    out << ",\"svd_cutoff\":\"" << json_escape(c.svd_cutoff) << "\"";
    out << ",\"expectation_engine\":\"" << json_escape(c.expectation_engine) << "\"";
    out << ",\"two_qubit_apply\":\"" << json_escape(c.two_qubit_apply) << "\"";
    if (c.threads > 0) {
        out << ",\"threads\":" << c.threads;
    }
    out << ",\"optimizer\":\"" << json_escape(c.optimizer) << "\"";
    out << ",\"profile_conversion\":" << (c.profile_conversion ? "true" : "false");
    out << ",\"fuse_single_qubit\":" << (c.fuse_single_qubit ? "true" : "false");
    out << ",\"absorb_1q_into_2q\":" << (c.absorb_1q_into_2q ? "true" : "false");
    out << ",\"check_ref\":" << (c.check_ref ? "true" : "false");
    out << ",\"check_ref_max_qubits\":" << c.check_ref_max_qubits;
    out << ",\"max_statevector_qubits\":30";
    out << "}";
    return out.str();
}

static std::string build_command(
    const Config& cfg,
    const Case& c,
    const fs::path& stdout_file,
    const fs::path& stderr_file
) {
    std::vector<std::string> args = {
        cfg.python,
        cfg.baseline,
        "--mode", c.mode,
        "--output", c.output,
        "--workload", c.workload,
        "--nqubits", std::to_string(c.nqubits),
        "--optimizer", c.optimizer,
        "--observable", c.observable,
        "--svd-cutoff", c.svd_cutoff,
        "--expectation-engine", c.expectation_engine,
        "--two-qubit-apply", c.two_qubit_apply,
        "--max-statevector-qubits", "30",
    };

    if (c.ngates >= 0) {
        args.push_back("--ngates");
        args.push_back(std::to_string(c.ngates));
    }
    if (c.has_max_bond) {
        args.push_back("--max-bond");
        args.push_back(std::to_string(c.max_bond));
    }
    if (c.threads > 0) {
        args.push_back("--threads");
        args.push_back(std::to_string(c.threads));
    }
    if (c.check_ref) {
        args.push_back("--check-ref");
        args.push_back("--check-ref-max-qubits");
        args.push_back(std::to_string(c.check_ref_max_qubits));
    }
    if (c.profile_conversion) {
        args.push_back("--profile-conversion");
    }
    if (c.fuse_single_qubit) {
        args.push_back("--fuse-single-qubit");
    }
    if (c.absorb_1q_into_2q) {
        args.push_back("--absorb-1q-into-2q");
    }

    std::ostringstream cmd;
    cmd << "CUDA_VISIBLE_DEVICES='' ";
    if (c.threads > 0) {
        const std::string t = shell_quote(std::to_string(c.threads));
        cmd << "OMP_NUM_THREADS=" << t << ' ';
        cmd << "OPENBLAS_NUM_THREADS=" << t << ' ';
        cmd << "MKL_NUM_THREADS=" << t << ' ';
        cmd << "VECLIB_MAXIMUM_THREADS=" << t << ' ';
        cmd << "NUMEXPR_NUM_THREADS=" << t << ' ';
    }
    if (c.timeout_sec > 0) {
        cmd << "timeout " << c.timeout_sec << ' ';
    }
    if (fs::exists("/usr/bin/time")) {
        cmd << "/usr/bin/time -v ";
    }
    for (size_t i = 0; i < args.size(); ++i) {
        if (i) cmd << ' ';
        cmd << shell_quote(args[i]);
    }
    cmd << " > " << shell_quote(stdout_file.string());
    cmd << " 2> " << shell_quote(stderr_file.string());
    return cmd.str();
}

static std::string build_batch_command(
    const Config& cfg,
    const std::vector<Case>& cases,
    const fs::path& batch_file,
    const fs::path& stdout_file,
    const fs::path& stderr_file
) {
    int batch_threads = 0;
    for (const auto& c : cases) {
        if (c.timeout_sec > 0) {
            throw std::runtime_error("--batch does not support per-case timeout suites.");
        }
        if (batch_threads == 0) {
            batch_threads = c.threads;
        } else if (c.threads != batch_threads) {
            throw std::runtime_error(
                "--batch requires all cases to use the same thread count. "
                "Run the threads suite without --batch."
            );
        }
    }

    std::ostringstream cmd;
    cmd << "CUDA_VISIBLE_DEVICES='' ";
    if (batch_threads > 0) {
        const std::string t = shell_quote(std::to_string(batch_threads));
        cmd << "OMP_NUM_THREADS=" << t << ' ';
        cmd << "OPENBLAS_NUM_THREADS=" << t << ' ';
        cmd << "MKL_NUM_THREADS=" << t << ' ';
        cmd << "VECLIB_MAXIMUM_THREADS=" << t << ' ';
        cmd << "NUMEXPR_NUM_THREADS=" << t << ' ';
    }
    if (fs::exists("/usr/bin/time")) {
        cmd << "/usr/bin/time -v ";
    }
    cmd << shell_quote(cfg.python) << ' '
        << shell_quote(cfg.baseline) << " --batch-json "
        << shell_quote(batch_file.string());
    cmd << " > " << shell_quote(stdout_file.string());
    cmd << " 2> " << shell_quote(stderr_file.string());
    return cmd.str();
}

static void fill_result_fields(
    std::map<std::string, std::string>& row,
    const Case& c,
    int rc,
    const fs::path& stdout_file,
    const fs::path& stderr_file,
    const std::string& stdout_text,
    const std::string& stderr_text,
    bool batch_run
) {
    row["returncode"] = std::to_string(rc);
    row["batch_run"] = batch_run ? "True" : "False";
    row["used_usr_bin_time"] = fs::exists("/usr/bin/time") ? "True" : "False";
    row["stdout_log"] = stdout_file.string();
    row["stderr_log"] = stderr_file.string();
    row["check_ref"] = c.check_ref ? "True" : "False";

    row["elapsed_sec_internal"] = json_number(stdout_text, "elapsed_sec");
    row["stage_total_sec"] = json_number(stdout_text, "stage_total_sec");
    row["stage_set_threads_sec"] = json_number(stdout_text, "stage_set_threads_sec");
    row["stage_set_backend_sec"] = json_number(stdout_text, "stage_set_backend_sec");
    row["stage_build_circuit_sec"] = json_number(stdout_text, "stage_build_circuit_sec");
    row["stage_execute_sec"] = json_number(stdout_text, "stage_execute_sec");
    row["stage_qibo_to_quimb_sec"] =
        json_number(stdout_text, "stage_qibo_to_quimb_sec");
    row["stage_quimb_circuit_init_sec"] =
        json_number(stdout_text, "stage_quimb_circuit_init_sec");
    row["stage_gate_loop_sec"] = json_number(stdout_text, "stage_gate_loop_sec");
    row["stage_gate_metadata_sec"] =
        json_number(stdout_text, "stage_gate_metadata_sec");
    row["stage_gate_apply_sec"] = json_number(stdout_text, "stage_gate_apply_sec");
    row["stage_gate_apply_1q_sec"] =
        json_number(stdout_text, "stage_gate_apply_1q_sec");
    row["stage_gate_apply_2q_sec"] =
        json_number(stdout_text, "stage_gate_apply_2q_sec");
    row["stage_gate_apply_other_sec"] =
        json_number(stdout_text, "stage_gate_apply_other_sec");
    row["stage_gate_fusion_matrix_sec"] =
        json_number(stdout_text, "stage_gate_fusion_matrix_sec");
    row["stage_gate_absorb_matrix_sec"] =
        json_number(stdout_text, "stage_gate_absorb_matrix_sec");
    row["stage_expectation_contract_sec"] =
        json_number(stdout_text, "stage_expectation_contract_sec");
    row["stage_mps_batched_attempt_sec"] =
        json_number(stdout_text, "stage_mps_batched_attempt_sec");
    row["actual_gate_count"] = json_number(stdout_text, "actual_gate_count");
    row["gate_count_1q"] = json_number(stdout_text, "gate_count_1q");
    row["gate_count_2q"] = json_number(stdout_text, "gate_count_2q");
    row["gate_count_other"] = json_number(stdout_text, "gate_count_other");
    row["gate_count_measure"] = json_number(stdout_text, "gate_count_measure");
    row["gate_count_applied_1q"] =
        json_number(stdout_text, "gate_count_applied_1q");
    row["gate_count_applied_2q"] =
        json_number(stdout_text, "gate_count_applied_2q");
    row["gate_count_applied_other"] =
        json_number(stdout_text, "gate_count_applied_other");
    row["gate_count_absorbed_1q"] =
        json_number(stdout_text, "gate_count_absorbed_1q");
    row["expectation_value"] = json_number(stdout_text, "expectation_value");
    row["expectation_engine"] = json_string(stdout_text, "expectation_engine");
    if (row["expectation_engine"].empty()) {
        row["expectation_engine"] = c.expectation_engine;
    }
    row["two_qubit_apply"] = json_string(stdout_text, "two_qubit_apply");
    if (row["two_qubit_apply"].empty()) {
        row["two_qubit_apply"] = c.two_qubit_apply;
    }
    row["expectation_terms"] = json_number(stdout_text, "expectation_terms");
    row["expectation_engine_fallback"] =
        json_string(stdout_text, "expectation_engine_fallback");
    row["state_norm"] = json_number(stdout_text, "state_norm");
    row["qibo"] = json_string(stdout_text, "qibo");
    row["qibotn"] = json_string(stdout_text, "qibotn");
    row["qibojit"] = json_string(stdout_text, "qibojit");
    row["quimb"] = json_string(stdout_text, "quimb");
    row["cotengra"] = json_string(stdout_text, "cotengra");
    row["ref_elapsed_sec"] = json_number_after(stdout_text, "\"reference\"", "elapsed_sec");
    row["max_abs_error"] = json_number_after(stdout_text, "\"reference\"", "max_abs_error");
    row["allclose_1e-8"] = json_bool_after(stdout_text, "\"reference\"", "allclose_1e-8");
    row["expected_expectation_value"] =
        json_number_after(stdout_text, "\"reference\"", "expected_expectation_value");
    row["abs_error"] = json_number_after(stdout_text, "\"reference\"", "abs_error");
    row["isclose_1e-8"] = json_bool_after(stdout_text, "\"reference\"", "isclose_1e-8");

    if (batch_run) {
        row["batch_time_elapsed_wall"] =
            time_field(stderr_text, "Elapsed \\(wall clock\\) time");
        row["batch_max_rss_kb"] =
            time_field(stderr_text, "Maximum resident set size \\(kbytes\\)");
    } else {
        row["time_elapsed_wall"] = time_field(stderr_text, "Elapsed \\(wall clock\\) time");
        row["time_user_sec"] = time_field(stderr_text, "User time \\(seconds\\)");
        row["time_system_sec"] = time_field(stderr_text, "System time \\(seconds\\)");
        row["cpu_percent"] = time_field(stderr_text, "Percent of CPU this job got");
        row["max_rss_kb"] = time_field(stderr_text, "Maximum resident set size \\(kbytes\\)");
        row["minor_page_faults"] =
            time_field(stderr_text, "Minor \\(reclaiming a frame\\) page faults");
        row["major_page_faults"] =
            time_field(stderr_text, "Major \\(requiring I/O\\) page faults");
    }

    std::string status = "OK";
    if (rc != 0) {
        status = "FAIL";
    } else if (c.check_ref && c.output == "expectation" && row["isclose_1e-8"] != "true") {
        status = "WRONG";
    } else if (c.check_ref && c.output == "statevector" && row["allclose_1e-8"] != "true") {
        status = "WRONG";
    } else if (row["elapsed_sec_internal"].empty()) {
        status = "FAIL";
    }
    row["status"] = status;
}

static std::map<std::string, std::string> run_case(const Config& cfg, const Case& c, int index) {
    fs::create_directories(cfg.log_dir);
    const std::string ts = timestamp_now();
    std::ostringstream tag;
    tag << ts << "_" << index
        << "_" << sanitize(c.workload)
        << "_" << sanitize(c.output)
        << "_" << sanitize(c.mode)
        << "_n" << c.nqubits;
    if (c.ngates >= 0) tag << "_g" << c.ngates;
    tag << "_" << sanitize(c.observable);
    if (c.has_max_bond) tag << "_bond" << c.max_bond;
    tag << "_cut" << sanitize(c.svd_cutoff);
    if (c.threads > 0) tag << "_thr" << c.threads;
    tag << "_" << sanitize(c.optimizer);
    tag << "_" << sanitize(c.expectation_engine);
    if (c.fuse_single_qubit) tag << "_fuse1q";
    if (c.absorb_1q_into_2q) tag << "_absorb1q2q";

    const fs::path stdout_file = fs::path(cfg.log_dir) / (tag.str() + ".stdout.json");
    const fs::path stderr_file = fs::path(cfg.log_dir) / (tag.str() + ".stderr.log");
    const std::string cmd = build_command(cfg, c, stdout_file, stderr_file);

    std::cout << "Running: " << cmd << std::endl;
    int rc = 0;
    if (!cfg.dry_run) {
        rc = system_return_code(std::system(cmd.c_str()));
    }

    const std::string stdout_text = cfg.dry_run ? "" : read_file(stdout_file);
    const std::string stderr_text = cfg.dry_run ? "" : read_file(stderr_file);

    std::map<std::string, std::string> row;
    row["timestamp"] = ts;
    row["mode"] = c.mode;
    row["output"] = c.output;
    row["workload"] = c.workload;
    row["nqubits"] = std::to_string(c.nqubits);
    row["requested_ngates"] = c.ngates >= 0 ? std::to_string(c.ngates) : "";
    row["max_bond"] = c.has_max_bond ? std::to_string(c.max_bond) : "";
    row["svd_cutoff"] = c.svd_cutoff;
    row["observable"] = c.observable;
    row["threads"] = c.threads > 0 ? std::to_string(c.threads) : "";
    row["optimizer"] = c.optimizer;
    row["expectation_engine"] = c.expectation_engine;
    row["two_qubit_apply"] = c.two_qubit_apply;
    if (cfg.dry_run) {
        row["returncode"] = "0";
        row["batch_run"] = "False";
        row["used_usr_bin_time"] = fs::exists("/usr/bin/time") ? "True" : "False";
        row["stdout_log"] = stdout_file.string();
        row["stderr_log"] = stderr_file.string();
        row["check_ref"] = c.check_ref ? "True" : "False";
        row["status"] = "DRYRUN";
    } else {
        fill_result_fields(row, c, rc, stdout_file, stderr_file, stdout_text, stderr_text, false);
    }

    if (!cfg.dry_run) {
        append_csv(fs::path(cfg.log_dir) / "summary.csv", row);
        append_jsonl(fs::path(cfg.log_dir) / "summary.jsonl", row);
    }
    return row;
}

static std::vector<std::string> jsonl_objects(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find('{');
        if (pos != std::string::npos) {
            out.push_back(line.substr(pos));
        }
    }
    return out;
}

static std::map<std::string, std::string> base_row_for_case(
    const std::string& ts,
    const Case& c
) {
    std::map<std::string, std::string> row;
    row["timestamp"] = ts;
    row["mode"] = c.mode;
    row["output"] = c.output;
    row["workload"] = c.workload;
    row["nqubits"] = std::to_string(c.nqubits);
    row["requested_ngates"] = c.ngates >= 0 ? std::to_string(c.ngates) : "";
    row["max_bond"] = c.has_max_bond ? std::to_string(c.max_bond) : "";
    row["svd_cutoff"] = c.svd_cutoff;
    row["observable"] = c.observable;
    row["threads"] = c.threads > 0 ? std::to_string(c.threads) : "";
    row["optimizer"] = c.optimizer;
    row["expectation_engine"] = c.expectation_engine;
    row["two_qubit_apply"] = c.two_qubit_apply;
    return row;
}

static std::vector<std::map<std::string, std::string>> run_batch_cases(
    const Config& cfg,
    const std::vector<Case>& cases
) {
    fs::create_directories(cfg.log_dir);
    const std::string ts = timestamp_now();
    const fs::path batch_file = fs::path(cfg.log_dir) / (ts + "_batch_cases.jsonl");
    const fs::path stdout_file = fs::path(cfg.log_dir) / (ts + "_batch.stdout.jsonl");
    const fs::path stderr_file = fs::path(cfg.log_dir) / (ts + "_batch.stderr.log");

    if (!cfg.dry_run) {
        std::ofstream cases_out(batch_file);
        for (size_t i = 0; i < cases.size(); ++i) {
            cases_out << case_json(cases[i], static_cast<int>(i + 1)) << "\n";
        }
    }

    const std::string cmd = build_batch_command(cfg, cases, batch_file, stdout_file, stderr_file);
    std::cout << "Running batch: " << cmd << std::endl;

    int rc = 0;
    if (!cfg.dry_run) {
        rc = system_return_code(std::system(cmd.c_str()));
    }

    const std::string stdout_text = cfg.dry_run ? "" : read_file(stdout_file);
    const std::string stderr_text = cfg.dry_run ? "" : read_file(stderr_file);
    const auto json_lines = jsonl_objects(stdout_text);

    std::vector<std::map<std::string, std::string>> rows;
    for (size_t i = 0; i < cases.size(); ++i) {
        auto row = base_row_for_case(ts, cases[i]);
        if (cfg.dry_run) {
            row["returncode"] = "0";
            row["batch_run"] = "True";
            row["used_usr_bin_time"] = fs::exists("/usr/bin/time") ? "True" : "False";
            row["stdout_log"] = stdout_file.string();
            row["stderr_log"] = stderr_file.string();
            row["check_ref"] = cases[i].check_ref ? "True" : "False";
            row["status"] = "DRYRUN";
        } else {
            const std::string line = i < json_lines.size() ? json_lines[i] : "";
            fill_result_fields(row, cases[i], rc, stdout_file, stderr_file, line, stderr_text, true);
        }
        rows.push_back(row);
        if (!cfg.dry_run) {
            append_csv(fs::path(cfg.log_dir) / "summary.csv", row);
            append_jsonl(fs::path(cfg.log_dir) / "summary.jsonl", row);
        }
    }
    return rows;
}

static Case main_case(const Config& cfg, const std::string& obs) {
    Case c;
    c.nqubits = cfg.nqubits;
    c.ngates = cfg.ngates;
    c.observable = obs;
    c.max_bond = cfg.max_bond;
    c.svd_cutoff = cfg.svd_cutoff;
    c.threads = cfg.threads;
    c.optimizer = cfg.optimizer;
    c.expectation_engine = cfg.expectation_engine;
    c.two_qubit_apply = cfg.two_qubit_apply;
    c.profile_conversion = cfg.profile_conversion;
    c.fuse_single_qubit = cfg.fuse_single_qubit;
    c.absorb_1q_into_2q = cfg.absorb_1q_into_2q;
    return c;
}

static void add_correctness(const Config& cfg, std::vector<Case>& cases) {
    for (const auto& obs : cfg.observables) {
        Case small = main_case(cfg, obs);
        small.nqubits = 12;
        small.ngates = 200;
        small.check_ref = true;
        cases.push_back(small);

        Case medium = main_case(cfg, obs);
        medium.nqubits = 20;
        medium.ngates = 1000;
        medium.check_ref = true;
        cases.push_back(medium);
    }
}

static void add_final(const Config& cfg, std::vector<Case>& cases) {
    for (int r = 0; r < cfg.repeats; ++r) {
        (void)r;
        for (const auto& obs : cfg.observables) {
            cases.push_back(main_case(cfg, obs));
        }
    }
}

static void add_stability(const Config& cfg, std::vector<Case>& cases) {
    for (const auto& obs : cfg.observables) {
        for (int bond : cfg.bonds) {
            for (const auto& cutoff : cfg.cutoffs) {
                Case c = main_case(cfg, obs);
                c.max_bond = bond;
                c.svd_cutoff = cutoff;
                cases.push_back(c);
            }
        }
    }
}

static void add_threads(const Config& cfg, std::vector<Case>& cases) {
    for (const auto& obs : cfg.observables) {
        for (int threads : cfg.thread_list) {
            Case c = main_case(cfg, obs);
            c.threads = threads;
            cases.push_back(c);
        }
    }
}

static void add_optimizers(const Config& cfg, std::vector<Case>& cases) {
    for (const auto& obs : cfg.observables) {
        for (const auto& opt : cfg.optimizers) {
            Case c = main_case(cfg, obs);
            c.optimizer = opt;
            cases.push_back(c);
        }
    }
}

static void add_modes(const Config& cfg, std::vector<Case>& cases) {
    for (int n : {8, 10, 12}) {
        for (const auto& mode : {"tn", "mps"}) {
            Case c = main_case(cfg, "z0");
            c.nqubits = n;
            c.ngates = 200;
            c.mode = mode;
            c.has_max_bond = std::string(mode) == "mps";
            c.optimizer = "auto";
            c.check_ref = true;
            c.timeout_sec = 120;
            cases.push_back(c);
        }
    }
}

static void add_tn_timeout(const Config& cfg, std::vector<Case>& cases) {
    Case c = main_case(cfg, "z0");
    c.mode = "tn";
    c.has_max_bond = false;
    c.optimizer = "greedy";
    c.timeout_sec = cfg.timeout_sec;
    cases.push_back(c);
}

static std::vector<Case> build_suite(const Config& cfg) {
    std::vector<Case> cases;
    if (cfg.suite == "correctness") {
        add_correctness(cfg, cases);
    } else if (cfg.suite == "final") {
        add_final(cfg, cases);
    } else if (cfg.suite == "stability") {
        add_stability(cfg, cases);
    } else if (cfg.suite == "threads") {
        add_threads(cfg, cases);
    } else if (cfg.suite == "optimizer") {
        add_optimizers(cfg, cases);
    } else if (cfg.suite == "modes") {
        add_modes(cfg, cases);
    } else if (cfg.suite == "tn-timeout") {
        add_tn_timeout(cfg, cases);
    } else if (cfg.suite == "all") {
        add_correctness(cfg, cases);
        add_final(cfg, cases);
        add_stability(cfg, cases);
        add_threads(cfg, cases);
        add_optimizers(cfg, cases);
        add_modes(cfg, cases);
        add_tn_timeout(cfg, cases);
    } else {
        throw std::runtime_error("Unknown suite: " + cfg.suite);
    }
    return cases;
}

static void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "C++ runner for qtn_baseline.py. It does not implement simulation;\n"
        << "it only launches the Python baseline and records logs.\n\n"
        << "Options:\n"
        << "  --suite NAME          final|correctness|stability|threads|optimizer|modes|tn-timeout|all\n"
        << "  --baseline PATH       Python baseline script (default qtn_baseline.py)\n"
        << "  --python PATH         Python executable (default python)\n"
        << "  --log-dir DIR         Output directory (default logs_cpp)\n"
        << "  --nqubits N           Large workload qubits (default 30)\n"
        << "  --ngates N            Large workload gates (default 10000)\n"
        << "  --max-bond N          Main MPS max bond (default 64)\n"
        << "  --svd-cutoff X        Main SVD cutoff (default 1e-8)\n"
        << "  --threads N           Main thread count (default 8)\n"
        << "  --optimizer NAME      Main optimizer (default auto-hq)\n"
        << "  --expectation-engine  local|mps-batched|auto (default local)\n"
        << "  --two-qubit-apply     name|matrix for custom conversion (default name)\n"
        << "  --profile-conversion  Record detailed Qibo->quimb conversion timings\n"
        << "  --fuse-single-qubit   Fuse pending single-qubit gates before MPS apply\n"
        << "  --absorb-1q-into-2q  Absorb pending 1q gates into following 2q gates\n"
        << "  --repeats N           Final suite repeats (default 3)\n"
        << "  --observables LIST    Comma/space list (default z0,zz-chain)\n"
        << "  --bonds LIST          Stability bonds (default 8,16,32,64,128,256)\n"
        << "  --cutoffs LIST        Stability cutoffs (default 1e-6,1e-8,1e-10)\n"
        << "  --thread-list LIST    Thread sweep list (default 1,2,4,8,16,32)\n"
        << "  --optimizers LIST     Optimizer sweep list (default greedy,auto,auto-hq,optimal)\n"
        << "  --timeout-sec N       TN timeout seconds (default 300)\n"
        << "  --batch               Run all compatible cases in one Python process\n"
        << "  --dry-run             Print commands without running\n"
        << "  --help                Show this help\n";
}

int main(int argc, char** argv) {
    Config cfg;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto need_value = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else if (arg == "--suite") {
                cfg.suite = need_value(arg);
            } else if (arg == "--baseline") {
                cfg.baseline = need_value(arg);
            } else if (arg == "--python") {
                cfg.python = need_value(arg);
            } else if (arg == "--log-dir") {
                cfg.log_dir = need_value(arg);
            } else if (arg == "--nqubits") {
                cfg.nqubits = std::stoi(need_value(arg));
            } else if (arg == "--ngates") {
                cfg.ngates = std::stoi(need_value(arg));
            } else if (arg == "--max-bond") {
                cfg.max_bond = std::stoi(need_value(arg));
            } else if (arg == "--svd-cutoff") {
                cfg.svd_cutoff = need_value(arg);
            } else if (arg == "--threads") {
                cfg.threads = std::stoi(need_value(arg));
            } else if (arg == "--optimizer") {
                cfg.optimizer = need_value(arg);
            } else if (arg == "--expectation-engine") {
                cfg.expectation_engine = need_value(arg);
            } else if (arg == "--two-qubit-apply") {
                cfg.two_qubit_apply = need_value(arg);
            } else if (arg == "--repeats") {
                cfg.repeats = std::stoi(need_value(arg));
            } else if (arg == "--observables") {
                cfg.observables = split_strings(need_value(arg));
            } else if (arg == "--bonds") {
                cfg.bonds = split_ints(need_value(arg));
            } else if (arg == "--cutoffs") {
                cfg.cutoffs = split_strings(need_value(arg));
            } else if (arg == "--thread-list") {
                cfg.thread_list = split_ints(need_value(arg));
            } else if (arg == "--optimizers") {
                cfg.optimizers = split_strings(need_value(arg));
            } else if (arg == "--timeout-sec") {
                cfg.timeout_sec = std::stoi(need_value(arg));
            } else if (arg == "--batch") {
                cfg.batch = true;
            } else if (arg == "--profile-conversion") {
                cfg.profile_conversion = true;
            } else if (arg == "--fuse-single-qubit") {
                cfg.fuse_single_qubit = true;
            } else if (arg == "--absorb-1q-into-2q") {
                cfg.absorb_1q_into_2q = true;
            } else if (arg == "--dry-run") {
                cfg.dry_run = true;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (cfg.expectation_engine != "local" &&
            cfg.expectation_engine != "mps-batched" &&
            cfg.expectation_engine != "auto") {
            throw std::runtime_error(
                "--expectation-engine must be local, mps-batched, or auto"
            );
        }
        if (cfg.two_qubit_apply != "name" && cfg.two_qubit_apply != "matrix") {
            throw std::runtime_error("--two-qubit-apply must be name or matrix");
        }

        const auto cases = build_suite(cfg);
        std::cout << "suite=" << cfg.suite << "\n";
        std::cout << "log_dir=" << cfg.log_dir << "\n";
        std::cout << "batch=" << (cfg.batch ? "true" : "false") << "\n";
        std::cout << "expectation_engine=" << cfg.expectation_engine << "\n";
        std::cout << "two_qubit_apply=" << cfg.two_qubit_apply << "\n";
        std::cout << "profile_conversion=" << (cfg.profile_conversion ? "true" : "false") << "\n";
        std::cout << "fuse_single_qubit=" << (cfg.fuse_single_qubit ? "true" : "false") << "\n";
        std::cout << "absorb_1q_into_2q=" << (cfg.absorb_1q_into_2q ? "true" : "false") << "\n";
        std::cout << "cases=" << cases.size() << "\n";
        std::vector<std::map<std::string, std::string>> rows;
        if (cfg.batch) {
            rows = run_batch_cases(cfg, cases);
        } else {
            for (size_t i = 0; i < cases.size(); ++i) {
                rows.push_back(run_case(cfg, cases[i], static_cast<int>(i + 1)));
            }
        }
        for (const auto& row : rows) {
            std::cout << "[" << map_get(row, "status") << "] "
                      << map_get(row, "mode") << " n=" << map_get(row, "nqubits")
                      << " obs=" << map_get(row, "observable")
                      << " engine=" << map_get(row, "expectation_engine")
                      << " elapsed=" << map_get(row, "elapsed_sec_internal")
                      << " q2q=" << map_get(row, "stage_qibo_to_quimb_sec")
                      << " apply=" << map_get(row, "stage_gate_apply_sec")
                      << " exp=" << map_get(row, "stage_expectation_contract_sec")
                      << " wall=" << (
                          !map_get(row, "time_elapsed_wall").empty()
                              ? map_get(row, "time_elapsed_wall")
                              : map_get(row, "batch_time_elapsed_wall")
                      )
                      << " rss=" << (
                          !map_get(row, "max_rss_kb").empty()
                              ? map_get(row, "max_rss_kb")
                              : map_get(row, "batch_max_rss_kb")
                      )
                      << "\n";
        }
        if (!cfg.dry_run) {
            std::cout << "summary: " << (fs::path(cfg.log_dir) / "summary.csv") << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        std::cerr << "Run with --help for usage.\n";
        return 2;
    }
    return 0;
}
