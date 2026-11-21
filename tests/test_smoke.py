"""Fast CPU smoke tests for the shared code paths."""
import sys

import pytest
import torch

from bptt import BPTT
from dataset import MultiSubjectDataset
from schemes.util import get_scheme_by_name

DATA = "./data/lorenz63/3params64sub/noisy.pt"


def test_scheme_lookup():
    for name in ("projection", "baseline"):
        assert callable(get_scheme_by_name(name))
    with pytest.raises(NotImplementedError):
        get_scheme_by_name("nonexistent")


def test_dataset_shapes():
    ds = MultiSubjectDataset(DATA, seq_len=10, size=200, subjects_per_batch=4,
                             num_workers=0, device="cpu")
    x, y, subject = ds[0]
    assert x.shape == y.shape == (10, ds.get_data().shape[-1])
    assert 0 <= int(subject) < ds.num_subjects


@pytest.mark.parametrize("scheme", ["projection", "baseline"])
def test_tiny_training_run(scheme, tmp_path, monkeypatch):
    """Two epochs end to end must run and produce a finite loss."""
    import main

    argv = ["main.py", "--data_path", DATA, "--train_set_size", "200", "--seq_len", "10",
            "--num_epochs", "2", "--batch_size", "32", "--batches_per_epoch", "2",
            "--hidden_size", "8", "--num_individual_params", "2",
            "--hierarchisation_scheme", scheme, "--save_path", str(tmp_path),
            "--metrics", "--plots", "--num_workers", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    args = main.handle_defaults(main.get_device(main.parse_args()))
    dataset = main.get_dataset(args)
    trainer = BPTT(args, dataset)
    trainer.train()
    assert not any(torch.isnan(p).any() for p in trainer.model.parameters())
