"""Shared pytest fixtures for card tier migration tests."""

from __future__ import annotations

import pandas as pd
import pytest

from card.data import generate_portfolio, load_portfolio
from card.modeling import load_model_bundle


@pytest.fixture(scope="session")
def portfolio_frame() -> pd.DataFrame:
    return load_portfolio()


@pytest.fixture(scope="session")
def bundle() -> dict:
    return load_model_bundle()
