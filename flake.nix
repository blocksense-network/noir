{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    verus-lib = {
      url = "github:Aristotelis2002/verus-lib";
      flake = false;
    };
  };

  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    fenix,
    verus-lib,
    ...
  }: let
    system = "x86_64-linux";
    venir-toolchain = fenix.packages.${system}.fromToolchainFile {
      file = ./venir-toolchain.toml;
      sha256 = "sha256-e4mlaJehWBymYxJGgnbuCObVlqMlQSilZ8FljG9zPHY=";
    };
  in
    flake-parts.lib.mkFlake {inherit inputs;} (
      let
        toolchain = with inputs.fenix.packages;
        with latest;
          combine [
            cargo
            clippy
            rust-analyzer
            rust-src
            rustc
            rustfmt
          ];
      in {
        systems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];

        perSystem = {
          pkgs,
          inputs',
          self',
          ...
        }: let
          toolchain = with inputs'.fenix.packages;
          with latest;
            combine [
              cargo
              clippy
              rust-analyzer
              rust-src
              rustc
              rustfmt
            ];
        in {
          legacyPackages.rustToolchain = toolchain;

          devShells.default = import ./shell.nix {inherit pkgs self' venir-toolchain verus-lib;};

          packages.default = import ./noir.nix {
            inherit pkgs;
            rust-toolchain = toolchain;
          };
        };
      }
    );
}
