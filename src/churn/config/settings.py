"""Central, typed configuration for the churn platform.

Every path, seed and constant lives in config.yaml. Other modules call
get_settings() instead of hard-coding values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# settings.py -> config -> churn -> src -> <project root>
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_root() -> Path:
    """Return the project root, allowing an environment-variable override."""
    env_root = os.environ.get("CHURN_PROJECT_ROOT")
    return Path(env_root).resolve() if env_root else _DEFAULT_ROOT


@dataclass(frozen=True)
class DataSettings:
    raw_filename: str
    target_column: str
    id_column: str
    positive_label: str


@dataclass(frozen=True)
class SplitSettings:
    train_size: float
    validation_size: float
    test_size: float

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")


@dataclass(frozen=True)
class MlflowSettings:
    experiment_name: str
    db_file: str


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    file: str


@dataclass(frozen=True)
class ThresholdSettings:
    recall_target: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 < self.recall_target <= 1.0:
            raise ValueError(f"recall_target must be in (0, 1], got {self.recall_target}")


@dataclass(frozen=True)
class BusinessSettings:
    """Retention economics for the business-cost threshold.

    The defaults are PLACEHOLDER assumptions, not company figures.
    """
    customer_value: float = 500.0
    offer_cost: float = 50.0
    save_rate: float = 0.30
    currency: str = "units"

    def __post_init__(self) -> None:
        if self.customer_value <= 0:
            raise ValueError(f"customer_value must be positive, got {self.customer_value}")
        if self.offer_cost < 0:
            raise ValueError(f"offer_cost cannot be negative, got {self.offer_cost}")
        if not 0.0 < self.save_rate <= 1.0:
            raise ValueError(f"save_rate must be in (0, 1], got {self.save_rate}")


@dataclass(frozen=True)
class Settings:
    project_name: str
    random_seed: int
    cv_splits: int
    root: Path
    data: DataSettings
    split: SplitSettings
    mlflow: MlflowSettings
    logging: LoggingSettings
    threshold: ThresholdSettings
    business: BusinessSettings

    # ---- data folders -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_dir / self.data.raw_filename

    # ---- artifact folders ---------------------------------------------
    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def tuned_models_dir(self) -> Path:
        return self.models_dir / "tuned"

    @property
    def candidates_dir(self) -> Path:
        return self.models_dir / "candidates"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_dir / "plots"

    @property
    def eda_dir(self) -> Path:
        return self.artifacts_dir / "eda"

    @property
    def shap_dir(self) -> Path:
        return self.artifacts_dir / "shap"

    @property
    def segmentation_dir(self) -> Path:
        return self.artifacts_dir / "segmentation"

    # ---- logging and tracking -----------------------------------------
    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / self.logging.file

    @property
    def mlflow_tracking_uri(self) -> str:
        db_path = (self.root / self.mlflow.db_file).as_posix()
        return f"sqlite:///{db_path}"

    def ensure_directories(self) -> None:
        """Create every output folder if it does not exist yet."""
        for folder in (
            self.raw_dir, self.interim_dir, self.processed_dir,
            self.models_dir, self.metrics_dir, self.plots_dir,
            self.eda_dir, self.shap_dir, self.segmentation_dir,
            self.logs_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    """Load config.yaml once and return an immutable Settings object."""
    root = _resolve_root()
    path = Path(config_path) if config_path else root / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    threshold_raw = raw.get("threshold") or {}
    business_raw = raw.get("business") or {}
    defaults = BusinessSettings()

    return Settings(
        project_name=raw["project"]["name"],
        random_seed=int(raw["project"]["random_seed"]),
        cv_splits=int(raw["cv"]["n_splits"]),
        root=root,
        data=DataSettings(**raw["data"]),
        split=SplitSettings(**raw["split"]),
        mlflow=MlflowSettings(**raw["mlflow"]),
        logging=LoggingSettings(**raw["logging"]),
        threshold=ThresholdSettings(
            recall_target=float(threshold_raw.get("recall_target", 0.80)),
        ),
        business=BusinessSettings(
            customer_value=float(business_raw.get("customer_value", defaults.customer_value)),
            offer_cost=float(business_raw.get("offer_cost", defaults.offer_cost)),
            save_rate=float(business_raw.get("save_rate", defaults.save_rate)),
            currency=str(business_raw.get("currency", defaults.currency)),
        ),
    )