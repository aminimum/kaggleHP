from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    raw_dir: Path
    output_dir: Path
    train_filename: str
    test_filename: str
    submission_filename: str

    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.raw_dir / self.test_filename

    @property
    def submission_path(self) -> Path:
        return self.raw_dir / self.submission_filename


@dataclass
class ModelConfig:
    default: str


@dataclass
class AppConfig:
    data: DataConfig
    models: ModelConfig


def _resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_config(config_path: Path) -> AppConfig:
    """Load application configuration from a YAML file."""
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw_config: Dict[str, Any] = yaml.safe_load(fh)

    project_root = config_path.parent.parent

    data_cfg = raw_config.get("data", {})
    models_cfg = raw_config.get("models", {})

    data_config = DataConfig(
        raw_dir=_resolve_path(project_root, data_cfg.get("raw_dir", "data/raw")),
        output_dir=_resolve_path(project_root, data_cfg.get("output_dir", "data/outputs")),
        train_filename=data_cfg.get("train_filename", "train.csv"),
        test_filename=data_cfg.get("test_filename", "test.csv"),
        submission_filename=data_cfg.get("submission_filename", "sample_submission.csv"),
    )

    model_config = ModelConfig(default=models_cfg.get("default", "linear"))

    return AppConfig(data=data_config, models=model_config)
