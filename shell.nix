# stanm: The single purpose of this script is to bootstrap rustup.
{
  pkgs,
  inputs',
  ...
}: let
  rust = with inputs'.fenix.packages;
  with latest;
    combine [
      cargo
      clippy
      rust-analyzer
      rust-src
      rustc
      rustfmt
    ];
in
  pkgs.mkShell {
    packages = [
      rust
    ];
  }
