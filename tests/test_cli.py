"""Command-line safety boundaries do not require any forecast data."""
import sys
import pytest
from beyond_time_distance.cli import fit_main, evaluate_main


def test_fit_rejects_same_partition(monkeypatch, tmp_path):
    p = str(tmp_path / "train.npz")
    monkeypatch.setattr(sys, "argv", ["btd-fit", "--train", p, "--validation", p, "--output", str(tmp_path / "model")])
    with pytest.raises(SystemExit):
        fit_main()


def test_evaluate_preserves_existing_file(monkeypatch, tmp_path):
    p = tmp_path / "scores.json"
    p.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["btd-evaluate", "--model", "missing", "--test", "missing", "--output", str(p)])
    with pytest.raises(SystemExit):
        evaluate_main()
    assert p.read_text() == "keep"
