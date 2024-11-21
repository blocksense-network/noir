{
  pkgs,
  rust-toolchain,
  ...
}: let
  inherit (pkgs);

  fenixRustPlatform = pkgs.makeRustPlatform {
    cargo = rust-toolchain;
    rustc = rust-toolchain;
  };
in
  fenixRustPlatform.buildRustPackage rec {
    name = "noir";
    pname = name;
    version = "0.1.0";

    src = ./.;

    nativeBuildInputs = with pkgs; [git];

    MAGENTA = "\e[35m";
    DEFAULT_COLOR = "\e[39m";
    RUST_BACKTRACE = 1;

    cargoLock = {
      lockFile = ./Cargo.lock;

      # This is awful, use crane instead
      outputHashes = {
        "air-0.1.0" = "sha256-DziMs2hjVZJ/5lsoRAVcmbPwEbW7heyad6qq/Se+QVE=";
        "chumsky-0.8.0" = "sha256-TvITrQMJlaBWx2tayYMX8AcvV4i0fyxrveBSMVojPMk=";
        "clap-markdown-0.1.3" = "sha256-2vG7x+7T7FrymDvbsR35l4pVzgixxq9paXYNeKenrkQ=";
        "getopts-0.2.21" = "sha256-r9CiPUSsjhThK6RG3AvhfTjaXMex/VV7CbdLQIDMdTk=";
        "plonky2-0.2.0" = "sha256-2oheUUDu4ggNZEX9sF3Ef3PNrdFIUg5POeOFIEXEkUY=";
        "plonky2_u32-0.1.0" = "sha256-COTm1Fi90+vCnc1MnqyKh8/DVzo/B9VO2o0RQvE9/nM=";
        "runtime_tracing-0.5.12" = "sha256-RcKL8tmdnLaGnMjreBr1zi3MOW/WXdMOS6bU8qPjmVQ=";
        "smt2parser-0.6.1" = "sha256-AKBq8Ph8D2ucyaBpmDtOypwYie12xVl4gLRxttv5Ods=";
      };
    };

    preBuild = ''
      echo "DEBUG PRINT"
      ls -la .
    '';
  }
