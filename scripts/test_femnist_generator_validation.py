#!/usr/bin/env python3
"""Validate lightweight FEMNIST generator input boundaries."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

import generate_FEMNIST as femnist


class FakeFemnistDataset:
    def __init__(self, data, transform):
        self.data = data
        self.images = torch.zeros(1, 1, 1, 1)
        self.targets = torch.zeros(1, dtype=torch.long)

    def __len__(self):
        return len(self.data)


def main() -> int:
    original_check = femnist.check
    original_read_pickle = femnist.pd.read_pickle
    original_get_writer_id = femnist.get_writer_id
    original_dataset = femnist.FemnistDataset
    original_dataloader = femnist.torch.utils.data.DataLoader

    femnist.check = lambda *args, **kwargs: False
    femnist.pd.read_pickle = lambda meta_path: [("writer-0", [("image.png", "30")])]
    femnist.get_writer_id = lambda data, num_clients: ["writer-0"]
    femnist.FemnistDataset = FakeFemnistDataset
    femnist.torch.utils.data.DataLoader = lambda *args, **kwargs: [
        (torch.zeros(2, 1, 1, 1), torch.zeros(1, dtype=torch.long))
    ]

    try:
        with tempfile.TemporaryDirectory(prefix="pifca-femnist-") as tmpdir:
            try:
                femnist.generate_dataset(
                    dataset="FEMNIST",
                    meta_path="unused.pkl",
                    save_path=tmpdir,
                    num_clients=1,
                    num_classes=62,
                    niid=True,
                    balance=False,
                    partition=None,
                )
            except ValueError as exc:
                if "images" not in str(exc) or "targets" not in str(exc):
                    raise AssertionError(f"unexpected ValueError message: {exc}") from exc
            else:
                raise AssertionError("FEMNIST generator accepted mismatched images and targets")
    finally:
        femnist.check = original_check
        femnist.pd.read_pickle = original_read_pickle
        femnist.get_writer_id = original_get_writer_id
        femnist.FemnistDataset = original_dataset
        femnist.torch.utils.data.DataLoader = original_dataloader

    print("PIFCA_FEMNIST_GENERATOR_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
