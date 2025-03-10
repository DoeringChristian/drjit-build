{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
      git
      gcc13
      stdenv.cc.cc.lib
      zlib
      ninja
      cmake
      ccache
      gdb
      pkg-config
      embree
      # python311Full
      llvmPackages_19.clang
      cudaPackages.cudatoolkit
  ];

  languages = {
    python = {
      enable = true;
      version = "3.11";
      venv.enable = true;
      venv.requirements = ''
      pytest>=8.3.3,<9
      numpy>=2.1.2,<3
      tqdm>=4.67.1,<5
      '';
    };
  };

  scripts = {
    configure-mitsuba.exec = ''
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=on -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja -S mitsuba3/ -B mitsuba3/build-mitsuba
    '';

    build-mitsuba.exec = ''
    configure-mitsuba
    cd mitsuba3/build-mitsuba
    ninja
    '';

    test-mitsuba.exec = ''
    build-mitsuba
    cd mitsuba3/build-mitsuba
    pytest
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
