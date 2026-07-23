"""
Tests for model_bundle module.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


def create_dummy_model():
    """Create a minimal Keras model for testing."""
    import tensorflow as tf
    from tensorflow.keras import Input, layers
    
    inputs = Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    outputs = layers.Conv2D(1, (3, 3), padding='same', activation='linear')(x)
    model = tf.keras.Model(inputs, outputs, name='test-model')
    model.compile(optimizer='adam', loss='mse')
    return model


def create_dummy_stats():
    """Create dummy stats dict matching build_standardized_lazy output."""
    return {
        'CHL': np.array([0.5, 1.2]),
        'masked_CHL': np.array([0.3, 1.0]),
        'feat_stats': {
            'sst': [15.0, 5.0],
            'so': [35.0, 2.0],
            'u10': [0.0, 5.0],
        }
    }


def test_save_and_load_bundle(tmp_path):
    """Test saving and loading a model bundle."""
    from mindthegap.model_bundle import save_model_bundle, load_model_bundle
    
    # Create dummy model and stats
    model = create_dummy_model()
    stats = create_dummy_stats()
    metadata = {
        'region': 'Test Region',
        'train_year': 2015,
        'train_range': 3,
    }
    
    # Save bundle
    bundle_path = tmp_path / "test_bundle"
    bundle = save_model_bundle(
        model=model,
        bundle_path=bundle_path,
        stats=stats,
        metadata=metadata,
    )
    
    # Check files exist
    assert (bundle_path / "model.keras").exists()
    assert (bundle_path / "stats.json").exists()
    assert (bundle_path / "metadata.json").exists()
    
    # Load bundle
    loaded_bundle = load_model_bundle(bundle_path)
    
    # Verify model
    assert loaded_bundle.model.name == 'test-model'
    assert loaded_bundle.model.count_params() == model.count_params()
    
    # Verify stats
    np.testing.assert_array_equal(loaded_bundle.stats['CHL'], stats['CHL'])
    np.testing.assert_array_equal(loaded_bundle.stats['masked_CHL'], stats['masked_CHL'])
    assert loaded_bundle.stats['feat_stats']['sst'] == stats['feat_stats']['sst']
    
    # Verify metadata
    assert loaded_bundle.metadata['region'] == 'Test Region'
    assert loaded_bundle.metadata['train_year'] == 2015


def test_save_with_history(tmp_path):
    """Test saving bundle with training history."""
    from mindthegap.model_bundle import save_model_bundle, load_model_bundle
    
    model = create_dummy_model()
    stats = create_dummy_stats()
    history = {
        'loss': [0.5, 0.3, 0.2],
        'val_loss': [0.6, 0.4, 0.3],
        'mae': [0.4, 0.25, 0.18],
    }
    
    bundle_path = tmp_path / "bundle_with_history"
    bundle = save_model_bundle(
        model=model,
        bundle_path=bundle_path,
        stats=stats,
        history=history,
    )
    
    # Check history file exists
    assert (bundle_path / "history.json").exists()
    
    # Load and verify
    loaded = load_model_bundle(bundle_path)
    assert loaded.history['loss'] == [0.5, 0.3, 0.2]
    assert loaded.history['val_loss'] == [0.6, 0.4, 0.3]


def test_overwrite_protection(tmp_path):
    """Test that overwrite protection works."""
    from mindthegap.model_bundle import save_model_bundle
    
    model = create_dummy_model()
    stats = create_dummy_stats()
    bundle_path = tmp_path / "test_bundle"
    
    # Save once
    save_model_bundle(model=model, bundle_path=bundle_path, stats=stats)
    
    # Try to save again without overwrite=True
    with pytest.raises(FileExistsError):
        save_model_bundle(model=model, bundle_path=bundle_path, stats=stats)
    
    # Should work with overwrite=True
    save_model_bundle(
        model=model,
        bundle_path=bundle_path,
        stats=stats,
        overwrite=True
    )


def test_missing_required_stats_keys(tmp_path):
    """Test that saving fails with incomplete stats."""
    from mindthegap.model_bundle import save_model_bundle
    
    model = create_dummy_model()
    incomplete_stats = {'CHL': np.array([0.5, 1.2])}  # Missing masked_CHL, feat_stats
    
    with pytest.raises(ValueError, match="stats must contain keys"):
        save_model_bundle(
            model=model,
            bundle_path=tmp_path / "bad_bundle",
            stats=incomplete_stats,
        )


def test_load_nonexistent_bundle():
    """Test loading a bundle that doesn't exist."""
    from mindthegap.model_bundle import load_model_bundle
    
    with pytest.raises(FileNotFoundError):
        load_model_bundle("/nonexistent/path")


def test_load_incomplete_bundle(tmp_path):
    """Test loading a bundle missing required files."""
    from mindthegap.model_bundle import load_model_bundle
    
    # Create directory but no files
    bundle_path = tmp_path / "incomplete_bundle"
    bundle_path.mkdir()
    
    with pytest.raises(FileNotFoundError):
        load_model_bundle(bundle_path)


def test_bundle_info(tmp_path):
    """Test bundle.info() output."""
    from mindthegap.model_bundle import save_model_bundle
    
    model = create_dummy_model()
    stats = create_dummy_stats()
    metadata = {'region': 'Arabian Sea', 'train_year': 2015}
    history = {'loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}
    
    bundle = save_model_bundle(
        model=model,
        bundle_path=tmp_path / "info_test",
        stats=stats,
        metadata=metadata,
        history=history,
    )
    
    info_str = bundle.info()
    
    # Check that key information is present
    assert 'Model Bundle' in info_str
    assert 'test-model' in info_str
    assert 'Arabian Sea' in info_str
    assert '2015' in info_str
    assert 'CHL' in info_str
    assert 'val_loss' in info_str


def test_model_bundle_class_methods(tmp_path):
    """Test using ModelBundle class directly."""
    from mindthegap.model_bundle import ModelBundle
    
    model = create_dummy_model()
    stats = create_dummy_stats()
    
    # Test save
    bundle = ModelBundle.save(
        model=model,
        bundle_path=tmp_path / "class_test",
        stats=stats,
    )
    
    assert bundle.model is model
    assert bundle.bundle_path == tmp_path / "class_test"
    
    # Test load
    loaded = ModelBundle.load(tmp_path / "class_test")
    assert loaded.model.name == 'test-model'


def test_stats_numpy_conversion(tmp_path):
    """Test that numpy arrays in stats are properly saved and loaded."""
    from mindthegap.model_bundle import save_model_bundle, load_model_bundle
    
    model = create_dummy_model()
    stats = {
        'CHL': np.array([1.5, 2.5]),
        'masked_CHL': np.array([1.0, 2.0]),
        'feat_stats': {
            'var1': [np.float64(10.0), np.float64(2.0)],
            'var2': [5.0, 1.5],
        }
    }
    
    bundle_path = tmp_path / "numpy_test"
    save_model_bundle(model=model, bundle_path=bundle_path, stats=stats)
    
    # Check that JSON file is valid
    with open(bundle_path / "stats.json") as f:
        stats_json = json.load(f)
    
    # Verify types in JSON (should be lists/floats, not numpy)
    assert isinstance(stats_json['CHL'], list)
    assert isinstance(stats_json['CHL'][0], (int, float))
    
    # Load and verify numpy arrays are restored
    loaded = load_model_bundle(bundle_path)
    assert isinstance(loaded.stats['CHL'], np.ndarray)
    assert isinstance(loaded.stats['masked_CHL'], np.ndarray)
    np.testing.assert_array_equal(loaded.stats['CHL'], stats['CHL'])


def test_convenience_functions(tmp_path):
    """Test top-level convenience functions."""
    from mindthegap import save_model_bundle, load_model_bundle
    
    model = create_dummy_model()
    stats = create_dummy_stats()
    
    # These should work just like the class methods
    bundle = save_model_bundle(
        model=model,
        bundle_path=tmp_path / "convenience_test",
        stats=stats,
    )
    
    loaded = load_model_bundle(tmp_path / "convenience_test")
    assert loaded.model.name == 'test-model'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
