#!/usr/bin/env bash
set -euo pipefail

CXX_BIN="${CXX:-g++}"
OUT="${1:-tools/qtn_runner}"

"$CXX_BIN" -std=c++17 -O2 -Wall -Wextra -pedantic \
  tools/qtn_runner.cpp \
  -o "$OUT"

echo "built $OUT"
echo "tip: tools/build_cpp_tools.sh also builds the C++ log summarizer"
