{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
      git
      gcc13
      libgccjit
      zlib
      ninja
      cmake
      ccache
      gdb
      pkg-config
      embree
      python311Full
      llvmPackages_19.clang
      cudaPackages.cudatoolkit
  ];

  scripts = {
    configure-mitsuba.exec = ''
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=on -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja -S mitsuba3/ -B mitsuba3/build-mitsuba
    '';

    build-mitsuba.exec = ''
    configure-mitsuba
    cd mitsuba3/build-mitsuba
    ninja
    '';
  };

  enterShell = ''
  source mitsuba3/build-mitsuba/setpath.sh
  '';

  env = {
    CMAKE_CXX_COMPILER_LAUNCHER = "ccache";
  };

  # See full reference at https://devenv.sh/reference/options/
}
