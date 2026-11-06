#!/usr/bin/env bash
# Builds everything needed by run.py (macOS arm64; tested on macOS 26, Apple clang 16).
#   prerequisites: Homebrew python@3.13, cmake, yaml-cpp   (brew install python@3.13 cmake yaml-cpp)
# Result: .venv with pydrake 1.58.0 + qsim_cpp (pangtao22/quasistatic_simulator @ c917755,
# patched for Drake 1.58 by qsim_drake1.58.patch) on its import path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
EXT="$ROOT/external"
DRAKE_VER=1.58.0
DRAKE_SHA=e539e17b0f0e0a2f4110a488b451c8a48ecb28297cff1609ff14226d2f3153ef
QSIM_COMMIT=c9177559ab5393ba800ebd264b3d97e4f7113fe9
PY=/opt/homebrew/bin/python3.13

# 1. Python environment (pydrake from the pip wheel).
[ -d "$ROOT/.venv" ] || "$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
SITE="$("$ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# 2. Drake C++ headers. The pip wheel ships no headers; the binary tarball of the
#    same release has them. qsim_cpp is compiled against these headers but loads
#    the wheel's libdrake.so at runtime, so only one Drake lives in the process.
mkdir -p "$EXT"
cd "$EXT"
if [ ! -d drake ]; then
  curl -fsSLO "https://github.com/RobotLocomotion/drake/releases/download/v$DRAKE_VER/drake-$DRAKE_VER-mac-arm64.tar.gz"
  echo "$DRAKE_SHA  drake-$DRAKE_VER-mac-arm64.tar.gz" | shasum -a 256 -c -
  tar xzf "drake-$DRAKE_VER-mac-arm64.tar.gz"
fi

# 3. quasistatic_simulator (CQDC model), pinned + patched.
if [ ! -d quasistatic_simulator ]; then
  git clone -q https://github.com/pangtao22/quasistatic_simulator.git
  git -C quasistatic_simulator checkout -q "$QSIM_COMMIT"
  git -C quasistatic_simulator apply "$ROOT/qsim_drake1.58.patch"
fi

# 4. Build qsim_cpp. Work around Command Line Tools installs whose clang cannot
#    find libc++ headers by pointing it at the SDK's copy (only if needed).
CXXFLAGS_EXTRA=""
if ! echo '#include <cmath>' | c++ -std=c++20 -x c++ -fsyntax-only - 2>/dev/null; then
  CXXFLAGS_EXTRA="-stdlib++-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"
  echo "note: using SDK libc++ headers ($CXXFLAGS_EXTRA)"
fi
BUILD="$EXT/quasistatic_simulator/quasistatic_simulator_cpp/build"
cmake -S "$EXT/quasistatic_simulator/quasistatic_simulator_cpp" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$EXT/drake;/opt/homebrew" \
  -DQSIM_PYTHON_VERSION=3.13 -DPython3_EXECUTABLE="$PY" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS_EXTRA" -Wno-dev > /dev/null
cmake --build "$BUILD" --target qsim_cpp -j 8

# 5. Load the wheel's libdrake.so (not the tarball's) and put qsim_cpp on sys.path.
SO="$(ls "$BUILD"/bindings/qsim_cpp*.so)"
install_name_tool -delete_rpath "$EXT/drake/lib" "$SO" 2>/dev/null || true
install_name_tool -add_rpath "$SITE/pydrake/lib" "$SO" 2>/dev/null || true
codesign -f -s - "$SO" 2>/dev/null
echo "$BUILD/bindings" > "$SITE/qsim_cpp.pth"
"$ROOT/.venv/bin/python" -c "import pydrake.all, qsim_cpp; print('qsim_cpp OK')"
