import numpy as np
import pytest
from carcara.core.trainer import MACETrainer
from ase.io import read

@pytest.fixture
def setup_trainer():
    # Create a simple MACETrainer instance for testing
    trainer = MACETrainer(
        name="test_model",
        train_file="data/train.xyz",
        valid_file="data/valid.xyz",
        max_num_epochs=10,
        restart_latest=False,
        num_channels=16,
        max_L=0,
        num_interaction=2,
        correlation=2,
        eval_interval=2,
        batch_size=5,
        valid_batch_size=10
    )
    return trainer
