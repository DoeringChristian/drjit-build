{
  description = "Build environment for Dr.Jit and Mitsbua3";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    ...
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [];
          config = {
            allowUnfree = true;
            cudaSupport = true;
            config.cudaVersion = "12";
          };
        };
      in rec {
        devShells = with pkgs; rec {
          default = mkShell {
            buildInputs = [
              # Tools
              bear
              linuxPackages_latest.perf

              # Basics
              git
              gcc13
              zlib
              ninja
              cmake
              ccache
              pkg-config
              stdenv.cc.cc.lib
              gdb

              # CUDA
              cudatoolkit

              # LLVM
              llvmPackages_19.clang
              llvm.lib
              llvm.dev

              # Python
              python312Full
              uv

              # Mitsuba dependencies
              embree
            ];

            # DRJIT_LIBLLVM_PATH = "${pkgs.llvm.lib}/lib/libLLVM.so";
            NIX_ENFORCE_NO_NATIVE = null;

            shellHook = ''
              export FLAKE_ROOT="$PWD"
              export CUDA_PATH=${pkgs.cudatoolkit}

              export CC="${gcc13}/bin/gcc"
              export CXX="${gcc13}/bin/g++"
              export PATH="${gcc13}/bin:$PATH"

              export CUDA_PATH="${pkgs.cudatoolkit}"
              export CLANGD_CUDA_INCLUDE="${pkgs.cudatoolkit}"

              export LD_LIBRARY_PATH="/run/opengl-driver/lib:${zlib}/lib:${llvm.lib}/lib:${stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"

			  export CMAKE_CXX_COMPILER_LAUNCHER="ccache"

              if [ ! -d .venv ]; then
                python -m venv .venv
              fi

              source .venv/bin/activate
              pip install -q -r requirements.txt

              source mitsuba3/build-mitsuba/setpath.sh
            '';
          };
        };
      }
    );
}
