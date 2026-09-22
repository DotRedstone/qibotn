#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string csv_escape(const std::string& value) {
    bool needs_quote = false;
    for (char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
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
    return regex_first(text, std::regex("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)"));
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

static std::string stem_without_suffix(const fs::path& path, const std::string& suffix) {
    std::string name = path.filename().string();
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

static std::vector<fs::path> find_stdout_jsons(const std::vector<fs::path>& roots) {
    std::vector<fs::path> files;
    for (const auto& root : roots) {
        std::error_code ec;
        if (!fs::exists(root, ec)) {
            std::cerr << "warning: skipping missing path: " << root.string() << "\n";
            continue;
        }
        if (fs::is_regular_file(root, ec)) {
            if (root.filename().string().find(".stdout.json") != std::string::npos) {
                files.push_back(root);
            }
            continue;
        }
        for (const auto& entry : fs::recursive_directory_iterator(root, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file(ec)) {
                continue;
            }
            const auto name = entry.path().filename().string();
            if (name.size() >= 12 && name.find(".stdout.json") != std::string::npos) {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<std::string> fields() {
    return {
        "file",
        "mode",
        "output",
        "workload",
        "nqubits",
        "requested_ngates",
        "actual_gate_count",
        "observable",
        "max_bond",
        "svd_cutoff",
        "threads",
        "optimizer",
        "expectation_engine",
        "two_qubit_apply",
        "expectation_terms",
        "expectation_engine_fallback",
        "elapsed_sec",
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
        "expectation_value",
        "gate_count_1q",
        "gate_count_2q",
        "gate_count_other",
        "gate_count_measure",
        "gate_count_applied_1q",
        "gate_count_applied_2q",
        "gate_count_applied_other",
        "gate_count_absorbed_1q",
        "gate_count_pending_1q_groups_absorbed",
        "gate_count_absorbed_1q_original",
        "gate_count_flushed_1q_original",
        "gate_count_plain_2q",
        "gate_count_fused_2q",
        "state_norm",
        "reference_error",
        "reference_ok",
        "time_elapsed_wall",
        "time_user_sec",
        "time_system_sec",
        "cpu_percent",
        "max_rss_kb",
        "status",
    };
}

static std::map<std::string, std::string> summarize_one(const fs::path& stdout_path) {
    const std::string stdout_text = read_file(stdout_path);
    fs::path stderr_path = stdout_path;
    const std::string stem = stem_without_suffix(stdout_path, ".stdout.json");
    stderr_path.replace_filename(stem + ".stderr.log");
    const std::string stderr_text = fs::exists(stderr_path) ? read_file(stderr_path) : "";

    std::map<std::string, std::string> row;
    row["file"] = stdout_path.string();
    row["mode"] = json_string(stdout_text, "mode");
    row["output"] = json_string(stdout_text, "output");
    row["workload"] = json_string(stdout_text, "workload");
    row["nqubits"] = json_number(stdout_text, "nqubits");
    row["requested_ngates"] = json_number(stdout_text, "requested_ngates");
    row["actual_gate_count"] = json_number(stdout_text, "actual_gate_count");
    row["observable"] = json_string(stdout_text, "observable");
    row["max_bond"] = json_number(stdout_text, "max_bond");
    row["svd_cutoff"] = json_number(stdout_text, "svd_cutoff");
    row["threads"] = json_number(stdout_text, "threads_arg");
    row["optimizer"] = json_string(stdout_text, "contraction_optimizer");
    row["expectation_engine"] = json_string(stdout_text, "expectation_engine");
    row["two_qubit_apply"] = json_string(stdout_text, "two_qubit_apply");
    row["expectation_terms"] = json_number(stdout_text, "expectation_terms");
    row["expectation_engine_fallback"] =
        json_string(stdout_text, "expectation_engine_fallback");
    row["elapsed_sec"] = json_number(stdout_text, "elapsed_sec");
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
    row["expectation_value"] = json_number(stdout_text, "expectation_value");
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
    row["gate_count_pending_1q_groups_absorbed"] =
        json_number(stdout_text, "gate_count_pending_1q_groups_absorbed");
    row["gate_count_absorbed_1q_original"] =
        json_number(stdout_text, "gate_count_absorbed_1q_original");
    row["gate_count_flushed_1q_original"] =
        json_number(stdout_text, "gate_count_flushed_1q_original");
    row["gate_count_plain_2q"] =
        json_number(stdout_text, "gate_count_plain_2q");
    row["gate_count_fused_2q"] =
        json_number(stdout_text, "gate_count_fused_2q");
    row["state_norm"] = json_number(stdout_text, "state_norm");

    std::string reference_error = json_number_after(stdout_text, "\"reference\"", "abs_error");
    if (reference_error.empty()) {
        reference_error = json_number_after(stdout_text, "\"reference\"", "max_abs_error");
    }
    row["reference_error"] = reference_error;

    std::string reference_ok = json_bool_after(stdout_text, "\"reference\"", "isclose_1e-8");
    if (reference_ok.empty()) {
        reference_ok = json_bool_after(stdout_text, "\"reference\"", "allclose_1e-8");
    }
    row["reference_ok"] = reference_ok;

    row["time_elapsed_wall"] = time_field(stderr_text, "Elapsed \\(wall clock\\) time");
    row["time_user_sec"] = time_field(stderr_text, "User time \\(seconds\\)");
    row["time_system_sec"] = time_field(stderr_text, "System time \\(seconds\\)");
    row["cpu_percent"] = time_field(stderr_text, "Percent of CPU this job got");
    row["max_rss_kb"] = time_field(stderr_text, "Maximum resident set size \\(kbytes\\)");

    if (row["mode"].empty() || row["elapsed_sec"].empty()) {
        row["status"] = "FAIL";
    } else if (!row["reference_ok"].empty() && row["reference_ok"] != "true") {
        row["status"] = "WRONG";
    } else {
        row["status"] = "OK";
    }
    return row;
}

static void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " LOG_DIR [LOG_DIR ...] > summary.csv\n\n"
        << "C++ summarizer for qtn_baseline.py logs. It only parses JSON and\n"
        << "/usr/bin/time -v logs; it does not implement quantum simulation.\n";
}

int main(int argc, char** argv) {
    if (argc == 1) {
        usage(argv[0]);
        return 2;
    }

    std::vector<fs::path> roots;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
        roots.emplace_back(arg);
    }

    const auto names = fields();
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << names[i];
    }
    std::cout << "\n";

    for (const auto& file : find_stdout_jsons(roots)) {
        const auto row = summarize_one(file);
        for (size_t i = 0; i < names.size(); ++i) {
            if (i) {
                std::cout << ",";
            }
            const auto it = row.find(names[i]);
            if (it != row.end()) {
                std::cout << csv_escape(it->second);
            }
        }
        std::cout << "\n";
    }

    return 0;
}
