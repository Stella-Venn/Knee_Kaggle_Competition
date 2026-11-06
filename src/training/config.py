"""Training run configuration: one dataclass, saved to JSON alongside every
run's checkpoint/metrics -- [redacted: private project notes]'s "fix seeds, log configs... every run
should be reproducible from a saved config plus a git commit hash" working
agreement, made concrete rather than left as a reminder to self.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    seed: int = 0
    val_fold: int = 0
    batch_size: int = 4
    n_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    target_shape: Tuple[int, int, int] = (32, 256, 256)
    target_spacing: Tuple[float, float, float] = (3.0, 0.5, 0.5)
    backbone_channels: Tuple[int, ...] = (16, 32, 64, 128)
    hidden_dim: Optional[int] = None
    max_pos_weight: float = 50.0
    nan_pos_weight_fallback: float = 1.0
    device: str = "cpu"
    cache_dir: str = "data/processed/volume_cache"
    manifest_path: str = "data/processed/manifest/series_manifest.csv"
    labels_path: str = "data/processed/training_labels.npz"
    folds_path: str = "data/processed/cv_folds.npz"


def save_config(config: TrainConfig, path, extra: Optional[dict] = None):
    """Writes `config` (plus any `extra` top-level keys, e.g. git_commit,
    final metrics) to `path` as JSON. Tuple fields serialize as JSON arrays;
    `load_config` converts them back.
    """
    payload = asdict(config)
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


_TUPLE_FIELDS = {"target_shape", "target_spacing", "backbone_channels"}


def load_config(path) -> TrainConfig:
    """Inverse of save_config -- ignores any extra keys a save_config call
    added beyond TrainConfig's own fields (e.g. git_commit, final metrics),
    so a run's saved JSON can be read back as a plain reusable config.
    """
    with open(path) as f:
        payload = json.load(f)
    field_names = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in payload.items() if k in field_names}
    for name in _TUPLE_FIELDS:
        if name in kwargs and kwargs[name] is not None:
            kwargs[name] = tuple(kwargs[name])
    return TrainConfig(**kwargs)
