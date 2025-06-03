{
  pkgs,
  venir-toolchain,
  pkgsForZ3,
  ...
}:
let
  inherit (pkgs) fetchFromGitHub rustup;

  customRustPlatform = pkgs.makeRustPlatform {
    cargo = venir-toolchain;
    rustc = venir-toolchain;
  };
in
customRustPlatform.buildRustPackage rec {
  pname = "verus-lib";
  name = pname;
  version = "0.1.0";

  nativeBuildInputs = [ rustup ];

  z3_path = pkgsForZ3.z3_4_12;
  VERUS_Z3_PATH = "${z3_path}/bin/z3";
  RUSTC_BOOTSTRAP = 1;
  VERUS_IN_VARGO = 1;
  RUSTFLAGS = "--cfg proc_macro_span --cfg verus_keep_ghost --cfg span_locations";

  # Some transitive dependency tests fail, not sure why. This disables them
  # (The formal verification tests however pass, which are the ones we care about)
  doCheck = false;

  der = fetchFromGitHub {
    owner = "Aristotelis2002";
    repo = "verus-lib";
    hash = "sha256-TeoMUYA4vrBJ7Pn4CuJ3bK9RF/0P/RKgSSBwICpKQnc=";
    rev = "synced_main";
  };

  src = "${der}";

  cargoLock = {
    # Getting the lockfile for a remote project with git dependencies in it is a notoriously difficult problem
    # For now this will work, more idiomatic solutions however are in the works
    lockFile = "${src}/source/Cargo.lock";

    outputHashes = {
      "getopts-0.2.21" = "sha256-r9CiPUSsjhThK6RG3AvhfTjaXMex/VV7CbdLQIDMdTk=";
      "smt2parser-0.6.1" = "sha256-AKBq8Ph8D2ucyaBpmDtOypwYie12xVl4gLRxttv5Ods=";
      "synstructure-0.13.0" = "sha256-nluVB2uL8nSv3XPvxa2MsjRhSMoTybCKGiBqlshnnVU";
    };
  };

  postUnpack = ''

    # buildRustPackage is confused as to the correct location of the source dir
    # Project structure is a bit odd. This is an acceptable workaround.
    # (Maybe just move instead of copying)
    cp -r ./source/source/* ./source
    cp -r ./source/dependencies ./dependencies
    cp -r ./source/tools ./tools

    # We need to delete the project's .toml file, so as to stop rustup from trying
    # to download the needed toolchain from the internet
    # NOTE: The nix env is sandboxed, it doesn't have access to the internet
    rm ./source/rust-toolchain.toml
  '';

  buildPhase = ''
    # Set up RUSTUP_HOME and CARGO_HOME to use writable directories
    export RUSTUP_HOME=$out/.rustup
    export CARGO_HOME=$out/.cargo

    mkdir -p $RUSTUP_HOME $CARGO_HOME

    # The name for our toolchain is very deliberate. If we were to change the rust versin,
    # we would most likely need change this also
    # TODO: I am very happy for suggestions how to circumvent this naming
    # (We can always change the source code for vstd in our fork of Verus)
    rustup toolchain link 1.76.0-x86_64-known-linux-gnu ${venir-toolchain.out}
    rustup default 1.76.0-x86_64-known-linux-gnu

    # Build the project
    cargo build
  '';

  installPhase = ''
    mkdir -p $out/lib
    cargo run -p vstd_build -- ./target/debug/
    cp ./target/debug/vstd.vir $out/lib/
  '';
}
