# stanm: The single purpose of this script is to bootstrap rustup.
{
  pkgs,
  self',
  venir-toolchain,
  ...
}: let
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
      hash = "sha256-R1w8aFCNsDaSmVlGpVzZPv4Li691DjEQL02GMRGDYNM";
      rev = "857a75f056d23fbd6254dbb3d72de36e57178607";
    };

    cargoLock = {
      # Getting the lockfile for a remote project with git dependencies in it is a notoriously difficult problem
      # For now this will work, more idiomatic solutions however are in the works
      lockFile = "${src}/Cargo.lock";

      outputHashes = {
        "getopts-0.2.21" = "sha256-r9CiPUSsjhThK6RG3AvhfTjaXMex/VV7CbdLQIDMdTk=";
        "smt2parser-0.6.1" = "sha256-AKBq8Ph8D2ucyaBpmDtOypwYie12xVl4gLRxttv5Ods=";
        "air-0.1.0" = "sha256-+8rK0Liv+62jVT3BNp1g/9jdsc0OpmV7hpcITMn5mRY=";
      };
    };

    cargoHash = "sha256-ER9SPWnoSmDln/8nTMeojSYvK4HxQzFCR6C6nSxFpOM=";

    preFixup = ''
      patchelf --set-rpath "${venir-toolchain}/lib" "$out/bin/${binaryName}"
    '';
  }