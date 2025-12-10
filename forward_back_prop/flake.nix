{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system: f (import nixpkgs { inherit system; })
        );
      pythonForPkgs =
        pkgs:
        pkgs.python3.withPackages (
          ppkgs: with ppkgs; [
            numpy
            matplotlib
          ]
        );
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          python = pythonForPkgs pkgs;
        in
        {
          default = pkgs.mkShell {
            packages = [
              (pkgs.writeShellScriptBin "pycharm" "${pkgs.tmux}/bin/tmux new -d 'pycharm-professional .'")
              python
            ];
            shellHook = ''
              echo "Python interpreter: ${python}/bin/python"
              exec -l zsh
            '';
          };
        }
      );
    };
}
