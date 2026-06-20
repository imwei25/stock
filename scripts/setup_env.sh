#!/usr/bin/env bash
# Bootstrap the full stockpool dev environment.
#
# Installs (idempotent, safe to re-run):
#   * Python virtualenv at .venv (uses system python3 / python >=3.10)
#   * stockpool package + dev deps (pytest, etc.)
#   * Rust stable toolchain (via rustup; per-user install)
#   * maturin (PyO3 build tool, into .venv)
#   * rust/stockpool_ops crate (if present — lands in PR-2 of the Rust ops plan)
#
# Supports: Windows (Git Bash / MSYS), macOS, Linux.
#
# Usage:
#   bash scripts/setup_env.sh                # full install
#   bash scripts/setup_env.sh --skip-rust    # Python-only (factor ops fall back to pandas)
#   bash scripts/setup_env.sh --help

set -euo pipefail

# ─── color logging ──────────────────────────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
else
    GREEN='' YELLOW='' RED='' NC=''
fi
log()  { printf '%b[setup]%b %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%b[setup]%b %s\n' "$YELLOW" "$NC" "$*"; }
fail() { printf '%b[setup]%b %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

# ─── args ────────────────────────────────────────────────────────────────────
SKIP_RUST=0
for arg in "$@"; do
    case "$arg" in
        --skip-rust) SKIP_RUST=1 ;;
        --help|-h)
            sed -n '1,/^set -euo pipefail$/p' "$0" | grep -E '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) fail "Unknown arg: $arg (try --help)" ;;
    esac
done

# ─── OS detection ────────────────────────────────────────────────────────────
case "$(uname -s 2>/dev/null || echo "${OS:-}")" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT) OSTYPE=windows ;;
    Darwin)                          OSTYPE=macos ;;
    Linux)                           OSTYPE=linux ;;
    *) fail "Unsupported OS: $(uname -s 2>/dev/null || echo unknown)" ;;
esac
log "OS detected: $OSTYPE"

if [ "$OSTYPE" = "windows" ]; then
    VENV_BIN=".venv/Scripts"
    VENV_PY="$VENV_BIN/python.exe"
    VENV_PIP="$VENV_BIN/pip.exe"
    VENV_MATURIN="$VENV_BIN/maturin.exe"
else
    VENV_BIN=".venv/bin"
    VENV_PY="$VENV_BIN/python"
    VENV_PIP="$VENV_BIN/pip"
    VENV_MATURIN="$VENV_BIN/maturin"
fi

# ─── Python ──────────────────────────────────────────────────────────────────
log "Checking Python >= 3.10..."
PYTHON=""
for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PYTHON="$cand"
            break
        fi
    fi
done
[ -z "$PYTHON" ] && fail "No Python >= 3.10 found. Install from https://www.python.org/downloads/"
log "Found: $($PYTHON --version)"

# ─── venv ────────────────────────────────────────────────────────────────────
if [ ! -x "$VENV_PY" ]; then
    log "Creating .venv with $PYTHON..."
    "$PYTHON" -m venv .venv
else
    log ".venv exists ($("$VENV_PY" --version))"
fi

# ─── stockpool + dev deps ────────────────────────────────────────────────────
log "Upgrading pip + installing stockpool[dev] (editable)..."
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PIP" install -e ".[dev]" --quiet
log "stockpool installed"

# ─── Rust toolchain (optional but recommended) ───────────────────────────────
if [ "$SKIP_RUST" = "1" ]; then
    warn "--skip-rust set; not installing Rust. Hot factor ops will use the"
    warn "pandas fallback (~5-10x slower on universe=all)."
    log "Done."
    log "Try:  $VENV_PY -m pytest tests/ -q"
    exit 0
fi

log "Checking Rust toolchain..."
# Look in PATH first, then in the canonical rustup install dir.
if ! command -v cargo >/dev/null 2>&1; then
    if [ -x "$HOME/.cargo/bin/cargo" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi
if ! command -v cargo >/dev/null 2>&1; then
    log "Rust not found, installing rustup (stable toolchain)..."
    case "$OSTYPE" in
        windows)
            if command -v winget >/dev/null 2>&1; then
                winget install --id Rustlang.Rustup --silent \
                    --accept-source-agreements --accept-package-agreements
            else
                fail "winget unavailable; install rustup manually: https://rustup.rs/"
            fi
            export PATH="$HOME/.cargo/bin:$PATH"
            ;;
        macos|linux)
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
                | sh -s -- -y --default-toolchain stable --no-modify-path
            # shellcheck disable=SC1091
            . "$HOME/.cargo/env"
            ;;
    esac
fi
command -v cargo >/dev/null 2>&1 || fail "cargo still not found after install; check $HOME/.cargo/bin"
log "Rust: $(rustc --version)"

# ─── Windows: MSVC linker check (Rust needs it) ──────────────────────────────
if [ "$OSTYPE" = "windows" ]; then
    MSVC_OK=0
    for vs_dir in \
        "/c/Program Files/Microsoft Visual Studio/2022" \
        "/c/Program Files/Microsoft Visual Studio/2019" \
        "/c/Program Files (x86)/Microsoft Visual Studio/2022" \
        "/c/Program Files (x86)/Microsoft Visual Studio/2019"; do
        if [ -d "$vs_dir" ] && find "$vs_dir" -maxdepth 6 -name cl.exe 2>/dev/null | grep -q .; then
            MSVC_OK=1
            log "MSVC linker found under $vs_dir"
            break
        fi
    done
    if [ "$MSVC_OK" = "0" ]; then
        warn "Visual Studio with 'Desktop development with C++' workload NOT detected."
        warn "Rust on Windows requires the MSVC linker. Install one of:"
        warn "  - Visual Studio 2022 Community: https://visualstudio.microsoft.com/vs/community/"
        warn "  - Build Tools for VS 2022:      https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        warn "Select the 'Desktop development with C++' workload during install."
        warn "Continuing — maturin build below will fail clearly if MSVC is missing."
    fi
fi

# ─── maturin ────────────────────────────────────────────────────────────────
log "Installing maturin into .venv..."
"$VENV_PIP" install --upgrade maturin --quiet
log "maturin: $("$VENV_MATURIN" --version)"

# ─── build the Rust crate if it exists ──────────────────────────────────────
if [ -d "rust/stockpool_ops" ]; then
    log "Building rust/stockpool_ops crate (release mode)..."
    # maturin develop -m PATH/Cargo.toml builds from anywhere; clearer than cd.
    "$VENV_MATURIN" develop --release --manifest-path rust/stockpool_ops/Cargo.toml
    log "Rust crate installed into .venv"
else
    warn "rust/stockpool_ops/ not present (lands in PR-2 of rust-ops plan)."
    warn "When it lands, re-run this script or run:"
    warn "  $VENV_MATURIN develop --release --manifest-path rust/stockpool_ops/Cargo.toml"
fi

log "All done. Verify with:"
log "  $VENV_PY -m pytest tests/ -q"
