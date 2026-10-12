import unittest
from unittest.mock import MagicMock, patch
import torch.nn as nn

from app.visual_bge.eva_clip.hf_model import HFTextEncoder


class TestHFTextEncoderConfig(unittest.TestCase):
    @patch("app.visual_bge.eva_clip.hf_model.AutoTokenizer")
    @patch("app.visual_bge.eva_clip.hf_model.AutoModel")
    @patch("app.visual_bge.eva_clip.hf_model.AutoConfig")
    def test_encoder_decoder_true(
        self, mock_autoconfig, mock_automodel, mock_autotokenizer
    ):
        mock_config = MagicMock()
        mock_config.is_encoder_decoder = True
        mock_config.model_type = "bert"
        mock_config.hidden_size = 768
        mock_config.vocab_size = 30522
        mock_autoconfig.from_pretrained.return_value = mock_config

        mock_model_instance = MagicMock()
        mock_encoder = MagicMock(spec=nn.Module)
        mock_model_instance.encoder = mock_encoder
        mock_automodel.from_pretrained.return_value = mock_model_instance

        encoder = HFTextEncoder(
            model_name_or_path="dummy-model",
            output_dim=768,
            tokenizer_name="dummy-model",
        )

        mock_automodel.from_pretrained.assert_called_once_with("dummy-model")
        self.assertEqual(encoder.transformer, mock_encoder)

    @patch("app.visual_bge.eva_clip.hf_model.AutoTokenizer")
    @patch("app.visual_bge.eva_clip.hf_model.AutoModel")
    @patch("app.visual_bge.eva_clip.hf_model.AutoConfig")
    def test_encoder_decoder_false(
        self, mock_autoconfig, mock_automodel, mock_autotokenizer
    ):
        mock_config = MagicMock()
        mock_config.is_encoder_decoder = False
        mock_config.model_type = "bert"
        mock_config.hidden_size = 768
        mock_config.vocab_size = 30522
        mock_autoconfig.from_pretrained.return_value = mock_config

        mock_model_instance = MagicMock(spec=nn.Module)
        mock_automodel.from_pretrained.return_value = mock_model_instance

        encoder = HFTextEncoder(
            model_name_or_path="dummy-model",
            output_dim=768,
            tokenizer_name="dummy-model",
        )

        mock_automodel.from_pretrained.assert_called_once_with(
            "dummy-model", add_pooling_layer=False
        )
        self.assertEqual(encoder.transformer, mock_model_instance)

    @patch("app.visual_bge.eva_clip.hf_model.AutoTokenizer")
    @patch("app.visual_bge.eva_clip.hf_model.AutoModel")
    @patch("app.visual_bge.eva_clip.hf_model.AutoConfig")
    def test_encoder_decoder_missing_attribute(
        self, mock_autoconfig, mock_automodel, mock_autotokenizer
    ):
        mock_config = MagicMock(spec=["model_type", "hidden_size", "vocab_size"])
        mock_config.model_type = "bert"
        mock_config.hidden_size = 768
        mock_config.vocab_size = 30522
        mock_autoconfig.from_pretrained.return_value = mock_config

        mock_model_instance = MagicMock(spec=nn.Module)
        mock_automodel.from_pretrained.return_value = mock_model_instance

        encoder = HFTextEncoder(
            model_name_or_path="dummy-model",
            output_dim=768,
            tokenizer_name="dummy-model",
        )

        mock_automodel.from_pretrained.assert_called_once_with(
            "dummy-model", add_pooling_layer=False
        )
        self.assertEqual(encoder.transformer, mock_model_instance)
