{
  pkgs,
  self',
  venir-toolchain,
  ...
}:
let
  inherit (pkgs) lib rustPlatform fetchFromGitHub;

  customRustPlatform = pkgs.makeRustPlatform {
    cargo = venir-toolchain;
    rustc = venir-toolchain;
  };
in
customRustPlatform.buildRustPackage rec {
  pname = "Venir";
  name = pname;
  binaryName = "venir";
  version = "0.1.0";

  RUSTC_BOOTSTRAP = 1;

  doCheck = false;

  src = fetchFromGitHub {
    owner = "blocksense-network";
    repo = "Venir";
    hash = "sha256-wgP+VkY3wpR+5NIXoFlbIWDU3VQudkwuSdyDX9MUuwU=";
    rev = "15c520be78978cc99ac4ffc799764faa7bcb77b7";
  };

  cargoLock = {
    # Getting the lockfile for a remote project with git dependencies in it is a notoriously difficult problem
    # For now this will work, more idiomatic solutions however are in the works
    lockFile = "${src}/Cargo.lock";

    outputHashes = {
      "getopts-0.2.21" = "sha256-N/QJvyOmLoU5TabrXi8i0a5s23ldeupmBUzP8waVOiU=";
      "smt2parser-0.6.1" = "sha256-AKBq8Ph8D2ucyaBpmDtOypwYie12xVl4gLRxttv5Ods=";
      "air-0.1.0" = "sha256-TeoMUYA4vrBJ7Pn4CuJ3bK9RF/0P/RKgSSBwICpKQnc=";
    };
  };

  preFixup = ''
    patchelf --set-rpath "${venir-toolchain}/lib" "$out/bin/${binaryName}"
  '';
}
