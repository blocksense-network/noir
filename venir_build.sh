#!/usr/bin/env bash

set -e

function cleanup()
{
  # Cleanup
  echo "Cleaning up temporary build directory: $TMP_DIR"
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT

# Save the original directory
ORIGINAL_DIR="$(pwd)"
TMP_DIR=$(mktemp -d)
echo "Cloning Venir into: $TMP_DIR"
git clone https://github.com/blocksense-network/Venir.git "$TMP_DIR/Venir"

cd "$TMP_DIR/Venir"
export RUSTC_BOOTSTRAP=1

echo "Building project..."
cargo build

# Resolve binary path
BIN_PATH="$TMP_DIR/Venir/target/debug/venir"

# Detect Rust toolchain's lib directory
RUST_LIB_DIR="$(rustc --print sysroot)/lib"

# Confirm Rust lib dir exists
if [[ ! -d "$RUST_LIB_DIR" ]]; then
  echo "Error: Rust lib directory not found: $RUST_LIB_DIR"
  exit 1
fi

# Copy binary to bin/
mkdir -p "$ORIGINAL_DIR/bin"
cp "$BIN_PATH" "$ORIGINAL_DIR/bin/"

# Patch RPATH to point to the Rust toolchain's lib
echo "Patching binary RPATH to: $RUST_LIB_DIR"
patchelf --set-rpath "$RUST_LIB_DIR" "$ORIGINAL_DIR/bin/venir"

echo "Build complete. Binary is in $ORIGINAL_DIR/bin/"
echo "To be able to use Noir FV please export $ORIGINAL_DIR/bin/venir to your path"

