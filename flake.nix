{
  description = "Build environment for Dr.Jit and Mitsbua3";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    nixgl,
    ...
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [nixgl.overlay];
          config = {
            allowUnfree = true;
            cudaSupport = true;
            config.cudaVersion = "12";
          };
        };
      in rec {
        scripts = {
          configure-mitsuba = pkgs.writeShellScriptBin "configure-mitsuba" ''
            cd $FLAKE_ROOT
            cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=on -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja -S mitsuba3/ -B mitsuba3/build-mitsuba
          '';
          build-mitsuba = pkgs.writeShellScriptBin "build-mitsuba" ''
            configure-mitsuba
            cd $FLAKE_ROOT/mitsuba3/build-mitsuba
            ninja
          '';
          test-mitsuba = pkgs.writeShellScriptBin "test-mitsuba" ''
            build-mitsuba
            pip install -r $FLAKE_ROOT/requirements.txt
            cd $FLAKE_ROOT/mitsuba3/build-mitsuba
            export PYTHONPATH=python:$PYTHONPATH
            python -m pytest $@
          '';
          debug-mitsuba = pkgs.writeShellScriptBin "debug-mitsuba" ''
            build-mitsuba
            pip install -r $FLAKE_ROOT/requirements.txt
            cd $FLAKE_ROOT/mitsuba3/build-mitsuba
            export PYTHONPATH=python:$PYTHONPATH
            gdb --args python -m pytest $@
          '';

          configure-drjit = pkgs.writeShellScriptBin "configure-drjit" ''
            cd $FLAKE_ROOT
            cmake -DDRJIT_ENABLE_TESTS=on -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja -S mitsuba3/ext/drjit/ -B mitsuba3/build-drjit
          '';
          build-drjit = pkgs.writeShellScriptBin "build-drjit" ''
            configure-drjit
            cd $FLAKE_ROOT/mitsuba3/build-drjit
            ninja
          '';
          test-drjit = pkgs.writeShellScriptBin "test-drjit" ''
            build-drjit
            pip install -r $FLAKE_ROOT/requirements.txt
            cd $FLAKE_ROOT/mitsuba3/build-drjit
            export PYTHONPATH=python:$PYTHONPATH
            python -m pytest $@
          '';
          debug-drjit = pkgs.writeShellScriptBin "debug-drjit" ''
            build-drjit
            pip install -r $FLAKE_ROOT/requirements.txt
            cd $FLAKE_ROOT/mitsuba3/build-drjit
            export PYTHONPATH=python:$PYTHONPATH
            gdb --args python -m pytest $@
          '';
        };

        devShells = with pkgs; rec {
          default = mkShell {
            buildInputs = [
              # Utility scripts
              scripts.build-mitsuba
              scripts.configure-mitsuba
              scripts.test-mitsuba
              scripts.debug-mitsuba

              scripts.build-drjit
              scripts.configure-drjit
              scripts.test-drjit
              scripts.debug-drjit

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
              pkgs.nixgl.auto.nixGLNvidia
              pkgs.nixgl.auto.nixGLDefault
              # linuxPackages.nvidia_x11

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

            CMAKE_CXX_COMPILER_LAUNCHER = "ccache";
            # DRJIT_LIBLLVM_PATH = "${pkgs.llvm.lib}/lib/libLLVM.so";
            NIX_ENFORCE_NO_NATIVE = null;

            shellHook = ''
              export FLAKE_ROOT="$PWD"
              export CUDA_PATH=${pkgs.cudatoolkit}

              export CC="${gcc13}/bin/gcc"
              export CXX="${gcc13}/bin/g++"
              export PATH="${gcc13}/bin:$PATH"

              export LD_LIBRARY_PATH="$(nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
              export LD_LIBRARY_PATH="${llvm.lib}/lib:$LD_LIBRARY_PATH"
            '';
          };

          test = mkShell {
            buildInputs =
              [
              ]
              ++ default.buildInputs;
          };
        };
      }
    );
}
