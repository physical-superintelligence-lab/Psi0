{
  description = "Development shell for the psi0 Astro site";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f {
            pkgs = import nixpkgs { inherit system; };
          });
    in
    {
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            ffmpeg
            nodejs_22
            poppler_utils
          ];

          shellHook = ''
            echo "Node $(node --version)"
            echo "npm $(npm --version)"
            echo "Run: npm install"
            echo "Then: npm run dev"
          '';
        };
      });
    };
}
