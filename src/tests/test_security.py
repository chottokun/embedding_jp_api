import pytest
from fastapi.testclient import TestClient
from app.main import app
import logging
from unittest.mock import MagicMock, patch

def test_pii_redaction_in_logs(caplog):
    # Set caplog to capture error logs
    caplog.set_level(logging.ERROR)

    # raise_server_exceptions=False is required to test the global exception handler
    client = TestClient(app, raise_server_exceptions=False)

    # Mock model and its methods to trigger an unhandled exception
    mock_model = MagicMock()
    mock_model.lock = MagicMock()
    # Enter/Exit for the "with model.lock" statement
    mock_model.lock.__enter__ = MagicMock(return_value=None)
    mock_model.lock.__exit__ = MagicMock(return_value=None)

    mock_model.max_seq_length = 8192

    mock_tokenizer = MagicMock()
    mock_tokenizer.num_special_tokens_to_add.return_value = 2
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    mock_model.tokenizer = mock_tokenizer

    # Use a real supported model name from config/models.yml
    model_name = "cl-nagoya/ruri-v3-30m"

    with patch("app.main.get_model", return_value=mock_model):
        # Mock encode to raise an exception containing an email address
        with patch.object(mock_model, "encode", side_effect=Exception("Error for user@example.com")):
            response = client.post(
                "/v1/embeddings",
                json={"input": "some input", "model": model_name}
            )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}

    # Check logs for redaction
    found_pii = False
    found_redacted = False

    for record in caplog.records:
        message = record.getMessage()
        if "user@example.com" in message:
            found_pii = True
        if "[REDACTED]" in message:
            found_redacted = True

    assert not found_pii, "PII (email) should NOT be present in logs"
    assert found_redacted, "PII should be replaced with [REDACTED] in logs"
