from unittest.mock import patch, MagicMock
import torch
from app.models import get_torch_dtype, get_model, _model_cache, _model_lock


def test_get_torch_dtype_mappings():
    with patch("app.models.TORCH_DTYPE", "float16"):
        assert get_torch_dtype() == torch.float16

    with patch("app.models.TORCH_DTYPE", "fp16"):
        assert get_torch_dtype() == torch.float16

    with patch("app.models.TORCH_DTYPE", "bfloat16"):
        assert get_torch_dtype() == torch.bfloat16

    with patch("app.models.TORCH_DTYPE", "bf16"):
        assert get_torch_dtype() == torch.bfloat16

    with patch("app.models.TORCH_DTYPE", "float32"):
        assert get_torch_dtype() == torch.float32

    with patch("app.models.TORCH_DTYPE", "fp32"):
        assert get_torch_dtype() == torch.float32

    with patch("app.models.TORCH_DTYPE", "unknown"):
        assert get_torch_dtype() is None


def test_get_model_passes_model_kwargs_for_torch_dtype():
    with patch("app.models.TORCH_DTYPE", "bfloat16"):
        with patch("app.models.SentenceTransformer") as mock_st:
            mock_inst = MagicMock()
            mock_st.return_value = mock_inst
            with _model_lock:
                _model_cache.pop("cl-nagoya/ruri-v3-30m", None)

            m = get_model("cl-nagoya/ruri-v3-30m", device="cpu")
            assert m == mock_inst
            mock_st.assert_called_once()
            _, kwargs = mock_st.call_args
            assert kwargs.get("model_kwargs") == {"torch_dtype": torch.bfloat16}
