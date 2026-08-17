"""Tests for the session set-up entry point mtg.set_up."""

import mindthegap as mtg
from mindthegap import Options
from mindthegap import session


def test_set_up_returns_default_options():
    options = mtg.set_up(verbose=False)
    assert isinstance(options, Options)
    assert options.smoke_test is False


def test_set_up_passes_smoke_test_and_seed():
    options = mtg.set_up(smoke_test=True, seed=11, verbose=False)
    # smoke_test shrinks the gridder just like Options.default.
    assert options.smoke_test is True
    assert options.gridder.tile_size == (16, 16)
    assert options.gridder.time_chunk == 10
    assert options.resolved_seed() == 11


def test_set_up_equivalent_to_options_default():
    from_set_up = mtg.set_up(seed=5, verbose=False)
    from_default = Options.default(seed=5)
    assert from_set_up == from_default


def test_set_up_sets_tf_log_level():
    mtg.set_up(verbose=False)
    import os

    assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "2"


def test_configure_tensorflow_handles_missing_tf(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("no tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    n_gpus, version = session._configure_tensorflow(verbose=True)
    assert n_gpus is None and version is None
    assert "TensorFlow not available" in capsys.readouterr().out
