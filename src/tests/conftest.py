import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def reset_api_key():
    """
    Ensure API_KEY is None by default for all tests to maintain backward compatibility
    and prevent failures if an API_KEY is set in the environment.
    """
    with patch("app.main.API_KEY", None):
        yield
