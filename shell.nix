{
  pkgs,
  pkgsForZ3,
  self',
  venir-toolchain,
  verus-lib,
  ...
}:
let
  inherit (pkgs) lib stdenv mkShell;
  inherit (pkgs.darwin.apple_sdk) frameworks;
  venir = import ./derivation.nix { inherit pkgs self' venir-toolchain; };
  verus-std = import ./verusStd.nix {
    inherit
      pkgs
      self'
      venir-toolchain
      verus-lib
      pkgsForZ3
      ;
  };
in
mkShell {
  packages =
    [
      pkgs.alejandra
      pkgs.mdbook
      pkgsForZ3.z3_4_12
      venir
      self'.legacyPackages.rustToolchain
      verus-std
      # pkgs.rustfilt
    ]
    ++ lib.optionals stdenv.isDarwin [
      pkgs.libiconv
      frameworks.CoreServices
    ]
    ++ lib.optionals (!stdenv.isDarwin) [
      # `wrangler` is the Cloudflare CLI for deploying the build Noir docs
      # web-site. The package is currently marked as broken on macOS.
      pkgs.wrangler
    ];
  shellHook = ''
    export VERUS_Z3_PATH=$(which z3)
    export VARGO_TARGET_DIR="${verus-std}/lib/";
  '';
}
