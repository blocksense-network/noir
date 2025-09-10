{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-for-z3.url = "github:NixOS/nixpkgs/c792c60b8a97daa7efe41a6e4954497ae410e0c1";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    verus-lib = {
      url = "github:Aristotelis2002/verus-lib?ref=synced_main";
      flake = false;
    };
  };

  outputs =
    inputs@{
      nixpkgs,
      nixpkgs-for-z3,
      flake-parts,
      fenix,
      verus-lib,
      ...
    }:
    let
      system = "x86_64-linux";
      venir-toolchain = fenix.packages.${system}.fromToolchainFile {
        file = ./venir-toolchain.toml;
        sha256 = "sha256-yMuSb5eQPO/bHv+Bcf/US8LVMbf/G/0MSfiPwBhiPpk=";
      };
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      perSystem =
        {
          pkgs,
          inputs',
          self',
          ...
        }:
        {
          legacyPackages.rustToolchain =
            with inputs'.fenix.packages;
            with stable;
            combine [
              cargo
              clippy
              rust-analyzer
              rust-src
              rustc
              rustfmt
            ];
          devShells.default = import ./shell.nix {
            inherit
              pkgs
              self'
              venir-toolchain
              verus-lib
              ;
            pkgsForZ3 = inputs'.nixpkgs-for-z3.legacyPackages;
          };
        };
    };
}
