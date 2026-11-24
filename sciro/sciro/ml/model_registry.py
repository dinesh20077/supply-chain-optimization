"""
SCIRO Model Registry.

Handles model persistence (save/load), versioning, and metadata tracking.
Models are saved as pickle files in the data/models/ directory.
"""

import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict, List

from sciro.config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized model persistence and metadata manager."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = get_settings().DATA_DIR / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.models_dir / "registry.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if self._metadata_path.exists():
            with open(self._metadata_path, "r") as f:
                return json.load(f)
        return {"models": {}}

    def _save_metadata(self) -> None:
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def save_model(
        self,
        model: Any,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> str:
        """Save a trained model with metadata.

        Args:
            model: Trained model object (scikit-learn compatible).
            model_name: Identifier (e.g. 'demand_forecaster').
            metrics: Evaluation metrics dict.
            params: Training parameters / hyperparameters.
            description: Human-readable description.

        Returns:
            Path to saved model file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"
        filename = f"{model_name}_{version}.pkl"
        filepath = self.models_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        entry = {
            "model_name": model_name,
            "version": version,
            "filename": filename,
            "filepath": str(filepath),
            "saved_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "params": params or {},
            "description": description,
        }

        if model_name not in self._metadata["models"]:
            self._metadata["models"][model_name] = []
        self._metadata["models"][model_name].append(entry)
        self._save_metadata()

        logger.info("Saved model '%s' (%s) to %s", model_name, version, filepath)
        return str(filepath)

    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load a model by name (defaults to latest version).

        Args:
            model_name: Model identifier.
            version: Specific version string. If None, loads latest.

        Returns:
            Deserialized model object.

        Raises:
            FileNotFoundError: If model not found.
        """
        if model_name not in self._metadata["models"]:
            raise FileNotFoundError(f"No models found for '{model_name}'")

        entries = self._metadata["models"][model_name]
        if version:
            entry = next((e for e in entries if e["version"] == version), None)
            if not entry:
                raise FileNotFoundError(f"Version '{version}' not found for '{model_name}'")
        else:
            entry = entries[-1]  # latest

        filepath = Path(entry["filepath"])
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.info("Loaded model '%s' (%s)", model_name, entry["version"])
        return model

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with their latest version info."""
        result = []
        for name, entries in self._metadata["models"].items():
            latest = entries[-1]
            result.append({
                "model_name": name,
                "versions": len(entries),
                "latest_version": latest["version"],
                "latest_saved_at": latest["saved_at"],
                "latest_metrics": latest["metrics"],
                "description": latest["description"],
            })
        return result

    def get_model_info(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all version info for a specific model."""
        return self._metadata["models"].get(model_name, [])

    def delete_model(self, model_name: str, version: Optional[str] = None) -> None:
        """Delete a model version (or all versions if version is None)."""
        if model_name not in self._metadata["models"]:
            return

        if version:
            entries = self._metadata["models"][model_name]
            entry = next((e for e in entries if e["version"] == version), None)
            if entry:
                filepath = Path(entry["filepath"])
                if filepath.exists():
                    filepath.unlink()
                entries.remove(entry)
                if not entries:
                    del self._metadata["models"][model_name]
        else:
            for entry in self._metadata["models"][model_name]:
                filepath = Path(entry["filepath"])
                if filepath.exists():
                    filepath.unlink()
            del self._metadata["models"][model_name]

        self._save_metadata()
        logger.info("Deleted model '%s' (version=%s)", model_name, version or "ALL")
