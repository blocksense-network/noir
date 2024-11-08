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
        venir
        self'.legacyPackages.rustToolchain
        # pkgs.rustfilt
      ]
      ++ lib.optionals stdenv.isDarwin [
        pkgs.libiconv
        frameworks.CoreServices
      ];
  }
