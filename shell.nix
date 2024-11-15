# stanm: The single purpose of this script is to bootstrap rustup.
{
  pkgs,
  self',
  venir-toolchain,
  ...
}: let
  inherit (pkgs) lib stdenv mkShell;
  inherit (pkgs.darwin.apple_sdk) frameworks;
  venir = import ./derivation.nix {inherit pkgs self' venir-toolchain;};
in
  mkShell {
    packages =
      [
        pkgs.alejandra
        pkgs.z3_4_12
        venir
        self'.legacyPackages.rustToolchain
        # pkgs.rustfilt
      ]
      ++ lib.optionals stdenv.isDarwin [
        pkgs.libiconv
        frameworks.CoreServices
      ];
    shellHook = ''
      export VERUS_Z3_PATH=$(which z3)
      #export VARGO_TARGET_DIR="../verus-lib/source/target-verus/debug"
    '';
  }
