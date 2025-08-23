{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=32f313e49e42f715491e1ea7b306a87c16fe0388";
    oxalica-rust.url = "github:oxalica/rust-overlay";
  };
  outputs =
    {
      self,
      nixpkgs,
      oxalica-rust,
    }:
    let
      pythonPackage = "python312";
      rustVersion = "1.86.0";
      cudaSupport = true;

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system:
          f (
            import nixpkgs {
              inherit system;
              config.allowUnfree = true;
              config.cudaSupport = cudaSupport;
              config.cudaVersion = "12";
              overlays = [ oxalica-rust.overlays.default ];
            }
          )
        );
      pythonForPkgs =
        pkgs:
        pkgs.${pythonPackage}.withPackages (
          pythonPackages:
          with pythonPackages;
          [
            pandas polars
            matplotlib altair
            # pytorch-bin
            # tensorflow-bin
            notebook
            jupyterlab-lsp
            python-lsp-server
          ]
          ++ (gpuDependantPackages pkgs)
        );

      dependencies =
        pkgs: with pkgs; [
          nodePackages.vscode-langservers-extracted
          nodePackages.yaml-language-server
          nodePackages.bash-language-server
          nodePackages.unified-language-server
        ];

      mkLibraryPath =
        pkgs:
        with pkgs;
        lib.makeLibraryPath [
          # add other library packages here if needed
          stdenv.cc.cc # numpy (on which scenedetect depends) needs C libraries
          cudaPackages.cuda_nvrtc # libncrtc.so for cupy
        ];

      # torchPackages = pkgs:
      #   with pkgs.${pythonPackage}.pkgs;
      #   [ pytorch-bin ];
      #   if pkgs.config.cudaSupport
      #   then [ pytorchWithCuda (torchvision.override { torch = pytorchWithCuda; }) ]
      #   else [ pytorch torchvision ];
      gpuDependantPackages =
        pkgs:
        with pkgs.${pythonPackage}.pkgs;
        if pkgs.config.cudaSupport then
          [
            # tensorflowWithCuda
            # (opencv-python-headless.override { inherit opencv4; })
            cupy # Numpy + GPU support
          ]
          ++ (with pkgs.cudaPackages; [
            cuda_cudart
            # cuda_nvrtc
            cudnn
            cutensor
            nccl
            cusparselt
          ])
          ++ (with pkgs; [
            cudatoolkit
            libGLU
            libGL
          ])
        else
          [
            # tensorflowWithoutCuda
            # opencv-python-headless
            numpy
          ];
    in
    {
      devShells = forAllSystems (pkgs: {
        default =
          let
            python = pythonForPkgs pkgs;
            cudaSupport = pkgs.config.cudaSupport;
            # oxalica-override = pkgs.rust-bin.stable.${rustVersion}.default.override {
            oxalica-override = pkgs.rust-bin.nightly."2025-08-18".default.override {
              extensions = [ "rust-src" "clippy" "rust-analyzer" "rustfmt" ];
            };
          in
          pkgs.mkShell {
            inputsFrom = [ ];
            packages = [
              (pkgs.writeShellScriptBin "pycharm" "tmux new -d 'pycharm-professional $1'")
              python
              
              # Rust
              pkgs.llvmPackages.bintools
              pkgs.pkg-config
              oxalica-override
              pkgs.openssl
              # Since aliases don't work
              (pkgs.writeShellScriptBin "rustrover" "tmux new -d 'rust-rover $1'")
              pkgs.nodejs-slim  # For MCP servers
              pkgs.openssl
            ]
            ++ (dependencies pkgs);
            
            RUST_SRC_PATH = "${oxalica-override}/lib/rustlib/src/rust";

            # ${python}/bin/python -c "import tensorflow as tf; import torch; import logging; tf.get_logger().setLevel(logging.ERROR); print(f'Tensorflow GPU devices available: {len(tf.config.list_physical_devices(device_type=\'GPU\')) > 0}\nPytorch CUDA devices available: {torch.cuda.is_available()}')"
            shellHook = ''
              ${
                if cudaSupport then
                  ''
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${mkLibraryPath pkgs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
                    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"                                                   # For tensorflow with GPU support
                    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                    export EXTRA_CCFLAGS="-I/usr/include"
                  ''
                else
                  ""
              }


              export PYTHONPATH="${python}/${python.sitePackages}"

              echo "=== RUST ==="
              echo
              echo "Rust version: $(rustc --version)"
              echo "Cargo version: $(cargo --version)"
              echo "Rust toolchain location: ${oxalica-override}/bin"
              echo "RUST_SRC_PATH (stdlib location): $RUST_SRC_PATH"
              echo
              echo
  


              export RUSTFLAGS='-C target-cpu=native'
              export RUST_BACKTRACE=full

              echo "=== PYTHON ==="
              echo
              echo "Setting PYTHONPATH to ${python}/${python.sitePackages}"
              export PYTHONPATH="${python}/${python.sitePackages}"
              echo Running $(python --version) @ $(which python) ${if pkgs.config.cudaSupport then "with CUDA support" else ""}
              echo
              
              exec -l zsh
            '';
          };
      });
    };
}
