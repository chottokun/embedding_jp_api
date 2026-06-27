import pytest


@pytest.fixture(autouse=True)
def skip_auth_for_tests():
    from app import main

    original_api_key = main.API_KEY
    main.API_KEY = None
    yield
    main.API_KEY = original_api_key
