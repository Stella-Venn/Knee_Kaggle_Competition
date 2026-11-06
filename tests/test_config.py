"""Tests for src/training/config.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.config import TrainConfig, load_config, save_config


def test_save_and_load_round_trips(tmp_path):
    cfg = TrainConfig(seed=7, val_fold=2, batch_size=8, target_shape=(16, 64, 64))
    path = tmp_path / "config.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_save_config_includes_extra_fields(tmp_path):
    cfg = TrainConfig()
    path = tmp_path / "config.json"
    save_config(cfg, path, extra={"git_commit": "abc123", "final_val_auc": 0.61})
    import json
    payload = json.loads(path.read_text())
    assert payload["git_commit"] == "abc123"
    assert payload["final_val_auc"] == 0.61


def test_load_config_ignores_extra_keys(tmp_path):
    cfg = TrainConfig(seed=3)
    path = tmp_path / "config.json"
    save_config(cfg, path, extra={"git_commit": "abc123"})
    loaded = load_config(path)
    assert loaded == cfg  # extra key doesn't become a TrainConfig field or error


def test_tuple_fields_survive_json_round_trip(tmp_path):
    cfg = TrainConfig(backbone_channels=(8, 16, 32))
    path = tmp_path / "config.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.backbone_channels == (8, 16, 32)
    assert isinstance(loaded.backbone_channels, tuple)
