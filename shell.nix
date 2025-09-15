let
  pkgs = import <nixpkgs> { };
  python = pkgs.python3.withPackages
    (p: with p; [ mido pylsp-mypy numpy scipy matplotlib plotly ruff ]);
in pkgs.mkShell {
  name = "teslasynth-py";
  env = {
    # Don't create venv using uv
    UV_NO_SYNC = "1";

    # Force uv to use Python interpreter from venv
    UV_PYTHON = python;

    # Prevent uv from downloading managed Python's
    UV_PYTHON_DOWNLOADS = "never";
  };
  buildInputs = with pkgs; [ rosegarden alsa-utils uv python ];
}
