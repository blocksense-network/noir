{pkgs, ...}: let
  inherit (pkgs) rustPlatform;
in
  rustPlatform.buildRustPackage rec {
    pname = "Noir";
    name = pname;
    binaryName = "nargo";
    version = "unstable";

    RUSTC_BOOTSTRAP = 1;

    GIT_COMMIT = "false";
    GIT_DIRTY = "false";
    # RUST_BACKTRACE = 1;

    doCheck = false;

    src = ./.;

    cargoLock = {
      lockFile = "${src}/Cargo.lock";

      outputHashes = {
        "clap-markdown-0.1.3" = "sha256-2vG7x+7T7FrymDvbsR35l4pVzgixxq9paXYNeKenrkQ=";
        "plonky2-0.2.0" = "sha256-2oheUUDu4ggNZEX9sF3Ef3PNrdFIUg5POeOFIEXEkUY=";
        "plonky2_u32-0.1.0" = "sha256-COTm1Fi90+vCnc1MnqyKh8/DVzo/B9VO2o0RQvE9/nM=";
        "runtime_tracing-0.7.0" = "sha256-pSm+e2ly9dgK6T6ZwoJSMuD7726IwZtms+OzQffvTyU=";
      };
    };

    cargoHash = "";
  }
