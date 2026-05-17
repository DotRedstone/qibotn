#!/usr/bin/env bash
set -euo pipefail

CXX_BIN="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -pedantic)

"$CXX_BIN" "${CXXFLAGS[@]}" tools/qtn_runner.cpp -o tools/qtn_runner
"$CXX_BIN" "${CXXFLAGS[@]}" tools/qtn_summarize.cpp -o tools/qtn_summarize

echo "built tools/qtn_runner"
echo "built tools/qtn_summarize"
