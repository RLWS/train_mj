from __future__ import annotations

from mjai.cli import main


def test_cli_help_parser_registers_train_command(capsys):
    exit_code = main(["train", "--config", "configs/base.yaml"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "mjai command 'train' is registered" in captured.out
    assert "loaded config: configs/base.yaml" in captured.out
