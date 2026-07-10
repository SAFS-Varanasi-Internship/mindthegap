"""
Model bundle management for gap-filling workflows.

A model bundle packages together:
- Trained TensorFlow/Keras model
- Standardization statistics (mean, stdev)
- Metadata (region, dates, configuration)
- Training history

This enables reproducible gap-filling workflows by keeping all necessary
artifacts together in a single directory.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


class ModelBundle:
    """
    Container for a gap-filling model bundle.
    
    A bundle is a directory containing:
    - model.keras: the trained TensorFlow model
    - stats.json: standardization statistics
    - metadata.json: training configuration and provenance
    - history.json: training history (optional)
    
    Example
    -------
    >>> # Save a bundle after training
    >>> bundle = ModelBundle.save(
    ...     model=trained_model,
    ...     bundle_path="models/arabsea_2015",
    ...     stats=stats_dict,
    ...     metadata={"region": "Arabian Sea", "train_year": 2015}
    ... )
    >>> 
    >>> # Load the bundle later
    >>> bundle = ModelBundle.load("models/arabsea_2015")
    >>> model = bundle.model
    >>> stats = bundle.stats
    """
    
    def __init__(
        self,
        model: Any,
        stats: Dict[str, Any],
        metadata: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
        bundle_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize a ModelBundle.
        
        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model.
        stats : dict
            Standardization statistics with keys 'CHL', 'masked_CHL', 'feat_stats'.
        metadata : dict
            Training configuration and provenance information.
        history : dict, optional
            Training history from model.fit().
        bundle_path : str or Path, optional
            Path where the bundle is stored (if loaded from disk).
        """
        self.model = model
        self.stats = stats
        self.metadata = metadata
        self.history = history
        self.bundle_path = Path(bundle_path) if bundle_path else None
    
    @classmethod
    def save(
        cls,
        model: Any,
        bundle_path: Union[str, Path],
        stats: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        history: Optional[Any] = None,
        overwrite: bool = False,
    ) -> "ModelBundle":
        """
        Save a model bundle to disk.
        
        Creates a directory structure:
        bundle_path/
            model.keras         # TensorFlow model
            stats.json          # Standardization statistics
            metadata.json       # Configuration and provenance
            history.json        # Training history (optional)
        
        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model to save.
        bundle_path : str or Path
            Directory path where the bundle will be saved (created if needed).
        stats : dict
            Standardization statistics from build_standardized_lazy().
            Should contain 'CHL', 'masked_CHL', and 'feat_stats' keys.
        metadata : dict, optional
            Additional metadata (region, dates, config, etc.). If None, creates minimal metadata.
        history : tf.keras.callbacks.History or dict, optional
            Training history from model.fit(). Can be the History object or dict.
        overwrite : bool, default False
            If True, overwrite existing bundle. If False, raise error if bundle exists.
        
        Returns
        -------
        ModelBundle
            The saved bundle instance.
        
        Raises
        ------
        FileExistsError
            If bundle_path exists and overwrite=False.
        ValueError
            If stats is missing required keys.
        
        Examples
        --------
        >>> bundle = ModelBundle.save(
        ...     model=model,
        ...     bundle_path="models/arabsea_2015",
        ...     stats=stats_dict,
        ...     metadata={
        ...         "region": "Arabian Sea",
        ...         "train_year": 2015,
        ...         "train_range": 3,
        ...         "patch_size": (40, 56),
        ...         "zarr_source": "gcs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr"
        ...     },
        ...     history=history
        ... )
        """
        bundle_path = Path(bundle_path)
        
        # Check overwrite
        if bundle_path.exists() and not overwrite:
            raise FileExistsError(
                f"Bundle already exists at {bundle_path}. Use overwrite=True to replace."
            )
        
        # Validate stats
        required_keys = {'CHL', 'masked_CHL', 'feat_stats'}
        if not required_keys.issubset(stats.keys()):
            raise ValueError(
                f"stats must contain keys {required_keys}, got {set(stats.keys())}"
            )
        
        # Create directory
        bundle_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = bundle_path / "model.keras"
        model.save(str(model_file))
        
        # Save stats (convert numpy arrays to lists for JSON)
        stats_serializable = _make_json_serializable(stats)
        with open(bundle_path / "stats.json", "w") as f:
            json.dump(stats_serializable, f, indent=2)
        
        # Build metadata
        if metadata is None:
            metadata = {}
        
        # Add bundle version and model info
        metadata.setdefault("bundle_version", "1.0")
        metadata.setdefault("model_name", model.name if hasattr(model, 'name') else "unet")
        
        # Add model architecture summary
        try:
            import tensorflow as tf
            metadata["model_params"] = model.count_params()
            metadata["input_shape"] = [
                int(d) if d is not None else None 
                for d in model.input_shape[1:]
            ]
            metadata["output_shape"] = [
                int(d) if d is not None else None 
                for d in model.output_shape[1:]
            ]
        except Exception:
            pass  # Not critical if we can't extract model info
        
        # Save metadata
        with open(bundle_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save history if provided
        history_dict = None
        if history is not None:
            if hasattr(history, 'history'):
                history_dict = history.history
            elif isinstance(history, dict):
                history_dict = history
            
            if history_dict:
                history_serializable = _make_json_serializable(history_dict)
                with open(bundle_path / "history.json", "w") as f:
                    json.dump(history_serializable, f, indent=2)
        
        return cls(
            model=model,
            stats=stats,
            metadata=metadata,
            history=history_dict,
            bundle_path=bundle_path,
        )
    
    @classmethod
    def load(cls, bundle_path: Union[str, Path]) -> "ModelBundle":
        """
        Load a model bundle from disk.
        
        Parameters
        ----------
        bundle_path : str or Path
            Path to the bundle directory.
        
        Returns
        -------
        ModelBundle
            Loaded bundle with model, stats, metadata, and history.
        
        Raises
        ------
        FileNotFoundError
            If bundle_path does not exist or required files are missing.
        
        Examples
        --------
        >>> bundle = ModelBundle.load("models/arabsea_2015")
        >>> predictions = bundle.model.predict(X_test)
        >>> unstandardized = unstdize(predictions, bundle.stats['CHL'])
        """
        import tensorflow as tf
        
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found at {bundle_path}")
        
        # Load model
        model_file = bundle_path / "model.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model = tf.keras.models.load_model(str(model_file))
        
        # Load stats
        stats_file = bundle_path / "stats.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_file}")
        with open(stats_file) as f:
            stats = json.load(f)
        
        # Convert stats arrays back to numpy
        stats = _convert_to_numpy(stats)
        
        # Load metadata
        metadata_file = bundle_path / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Load history if present
        history_file = bundle_path / "history.json"
        history = None
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
        
        return cls(
            model=model,
            stats=stats,
            metadata=metadata,
            history=history,
            bundle_path=bundle_path,
        )
    
    def info(self) -> str:
        """
        Return a human-readable summary of the bundle.
        
        Returns
        -------
        str
            Formatted summary of bundle contents.
        """
        lines = []
        lines.append(f"Model Bundle: {self.bundle_path or '(not saved)'}")
        lines.append("=" * 60)
        
        # Model info
        lines.append("\nModel:")
        lines.append(f"  Name: {self.metadata.get('model_name', 'unknown')}")
        if 'model_params' in self.metadata:
            lines.append(f"  Parameters: {self.metadata['model_params']:,}")
        if 'input_shape' in self.metadata:
            lines.append(f"  Input shape: {self.metadata['input_shape']}")
        if 'output_shape' in self.metadata:
            lines.append(f"  Output shape: {self.metadata['output_shape']}")
        
        # Stats info
        lines.append("\nStandardization Statistics:")
        if 'CHL' in self.stats:
            chl_mean, chl_std = self.stats['CHL']
            lines.append(f"  CHL: mean={chl_mean:.4f}, std={chl_std:.4f}")
        if 'feat_stats' in self.stats:
            lines.append(f"  Features: {len(self.stats['feat_stats'])} variables")
            for name, (mean, std) in list(self.stats['feat_stats'].items())[:3]:
                lines.append(f"    {name}: mean={mean:.4f}, std={std:.4f}")
            if len(self.stats['feat_stats']) > 3:
                lines.append(f"    ... and {len(self.stats['feat_stats']) - 3} more")
        
        # Metadata
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                if key not in ['model_params', 'input_shape', 'output_shape', 
                              'model_name', 'bundle_version']:
                    lines.append(f"  {key}: {value}")
        
        # Training history
        if self.history:
            lines.append("\nTraining History:")
            if 'loss' in self.history:
                final_loss = self.history['loss'][-1]
                lines.append(f"  Final loss: {final_loss:.6f}")
            if 'val_loss' in self.history:
                final_val_loss = self.history['val_loss'][-1]
                lines.append(f"  Final val_loss: {final_val_loss:.6f}")
            if 'loss' in self.history:
                lines.append(f"  Epochs: {len(self.history['loss'])}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        path_str = str(self.bundle_path) if self.bundle_path else "(unsaved)"
        return f"ModelBundle(path={path_str})"


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and other non-JSON types to serializable types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def _convert_to_numpy(obj: Any) -> Any:
    """
    Recursively convert lists back to numpy arrays where appropriate.
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key in ['CHL', 'masked_CHL'] and isinstance(value, list):
                result[key] = np.array(value)
            else:
                result[key] = _convert_to_numpy(value)
        return result
    else:
        return obj


# Convenience functions for direct save/load
def save_model_bundle(
    model: Any,
    bundle_path: Union[str, Path],
    stats: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    history: Optional[Any] = None,
    overwrite: bool = False,
) -> ModelBundle:
    """
    Save a model bundle. Convenience wrapper for ModelBundle.save().
    
    See ModelBundle.save() for full documentation.
    """
    return ModelBundle.save(
        model=model,
        bundle_path=bundle_path,
        stats=stats,
        metadata=metadata,
        history=history,
        overwrite=overwrite,
    )


def load_model_bundle(bundle_path: Union[str, Path]) -> ModelBundle:
    """
    Load a model bundle. Convenience wrapper for ModelBundle.load().
    
    See ModelBundle.load() for full documentation.
    """
    return ModelBundle.load(bundle_path)
