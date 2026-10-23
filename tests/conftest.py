import pytest

from ml.train import train_model
from tests.helpers import make_dataset


@pytest.fixture(scope="session")
def trained():
    return train_model(make_dataset())
