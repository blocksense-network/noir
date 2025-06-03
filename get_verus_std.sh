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
echo "Cloning Verus-lib into: $TMP_DIR"
git clone https://github.com/Aristotelis2002/verus-lib.git -b synced_main "$TMP_DIR/Verus-lib"

cd "$TMP_DIR/Verus-lib"
chmod +x "$TMP_DIR/Verus-lib/source/tools/get-z3.sh"
sed -i '1s|.*|#!/usr/bin/env bash|' "$TMP_DIR/Verus-lib/source/tools/get-z3.sh"
${TMP_DIR}/Verus-lib/source/tools/get-z3.sh
cd "$TMP_DIR/Verus-lib/source"

export VERUS_Z3_PATH="$TMP_DIR/Verus-lib/z3"
export RUSTC_BOOTSTRAP=1
export VERUS_IN_VARGO=1
export RUSTFLAGS="--cfg proc_macro_span --cfg verus_keep_ghost --cfg span_locations"

echo "Building verus-lib..."
cargo build

echo "Building vstd.vir ..."
cd "$TMP_DIR/Verus-lib/source/"
cargo run -p vstd_build -- ./target/debug/
mkdir -p "$ORIGINAL_DIR/lib"
cp "$TMP_DIR/Verus-lib/source/target/debug/vstd.vir" "$ORIGINAL_DIR/lib/vstd.vir"

# Resolve binary path
export VERUS_STD_PATH="$ORIGINAL_DIR/lib/vstd.vir"

echo "Build completed. vstd.vir is located in $ORIGINAL_DIR/lib/"
echo "To be able to use Noir FV please export $ORIGINAL_DIR/lib/ as VARGO_TARGET_DIR"

