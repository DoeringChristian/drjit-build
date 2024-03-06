{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
        config.cudaSupport = true;
        config.cudaVersion = "12";
      };
      cuda-redist = with pkgs.cudaPackages; [
        cuda_cudart.dev # cuda_runtime.h
        cuda_cudart.lib
        cuda_cudart.static
        cuda_cccl.dev # <nv/target>
        libcublas.dev # cublas_v2.h
        libcublas.lib
        libcusolver.dev # cusolverDn.h
        libcusolver.lib
        libcusparse.dev # cusparse.h
        libcusparse.lib
        cudatoolkit
      ];
      cuda = pkgs.symlinkJoin {
        name = "cuda";
        paths = cuda-redist;
      };
      llvm-redist = with pkgs.llvmPackages_17; [
        llvm.lib
        llvm.dev
        libllvm
        libclang
      ];
      llvm-path = pkgs.symlinkJoin {
        name = "llvm";
        paths = llvm-redist;
      };
      backendStdenv = pkgs.cudaPackages.backendStdenv;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.gcc11
        ] ++ cuda-redist ++ llvm-redist;
        shellHook = ''
          export CUDA_HOME="${cuda}"
          export LIBRARY_PATH="${cuda}/lib/stubs:$LIBRARY_PATH"
          export CC="${backendStdenv.cc}/bin/cc"
          export CXX="${backendStdenv.cc}/bin/c++"
          export LD_LIBRARY_PATH="${backendStdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${cuda}/lib:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64"
          
          export DRJIT_LIBLLVM_PATH=${llvm-path}/lib/libLLVM-17.so
        '';
      };
    };
}
