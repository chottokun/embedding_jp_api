from .config import (
    EMBEDDING_MODELS,
    RERANK_MODELS,
    TORCH_DTYPE,
    LOGIT_GATE_MODELS,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import logging
import threading
from contextlib import nullcontext
from typing import Optional, Any
from PIL import Image
from unittest.mock import MagicMock
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_torch_dtype() -> Optional[torch.dtype]:
    """
    Parses TORCH_DTYPE configuration into a torch.dtype.
    Supports float16, bfloat16, and float32. Returns None if unset.
    """
    if not TORCH_DTYPE:
        return None
    dtype_str = TORCH_DTYPE.lower().strip()
    if dtype_str in {"float16", "fp16"}:
        return torch.float16
    elif dtype_str in {"bfloat16", "bf16"}:
        return torch.bfloat16
    elif dtype_str in {"float32", "fp32"}:
        return torch.float32
    return None


# --- Logit Gate Model Wrapper ---


class LogitGateModelWrapper:
    supports_audio: bool = False
    supports_video: bool = False

    def __init__(
        self, model_name: str, device: str = "cuda", dtype: Optional[torch.dtype] = None
    ):
        import threading

        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.lock = threading.RLock()
        self.tokenizer_lock = threading.RLock()

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        model_kwargs = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                trust_remote_code=True,
                **model_kwargs,
            )
        except Exception:
            # Fallback if sdpa is not supported
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **model_kwargs,
            )

        self.model.to(self.device)
        self.model.eval()


# --- Multimodal Model Wrapper ---


class EmbeddingGemma2Model:
    supports_multimodal: bool = True

    @property
    def supports_audio(self) -> bool:
        from .config import ENABLE_AUDIO_EMBEDDING

        return ENABLE_AUDIO_EMBEDDING

    @property
    def supports_video(self) -> bool:
        from .config import ENABLE_VIDEO_EMBEDDING

        return ENABLE_VIDEO_EMBEDDING

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        from .config import ENABLE_AUDIO_EMBEDDING

        self.device = (
            device
            if device and torch.cuda.is_available() and device.startswith("cuda")
            else "cpu"
        )
        self.lock = threading.RLock()
        self.tokenizer_lock = threading.RLock()
        self.max_seq_length = 8192

        if dtype == torch.float16:
            logging.warning(
                "float16 is not supported safely for EmbeddingGemma2. Falling back to bfloat16 or float32."
            )
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.dtype = dtype

        # Load with SentenceTransformer since it's the standard for embeddings.
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": self.dtype} if self.dtype else {}
        config_kwargs = {} if ENABLE_AUDIO_EMBEDDING else {"audio_config": None}

        # In sentence-transformers, model_kwargs can be passed to the AutoModel underlying.
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            trust_remote_code=True,
        )
        self.model.max_seq_length = self.max_seq_length
        self.tokenizer = self.model.tokenizer

    def encode(
        self, texts: list[str], prompt_name: str | None = None, **kwargs
    ) -> list[list[float]]:
        return self.encode_multimodal(
            [(t, None) for t in texts], prompt_name=prompt_name
        )

    def encode_multimodal(
        self,
        items: list[Any],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        processed_inputs = []
        for it in items:
            if isinstance(it, dict):
                processed_inputs.append(it)
            elif isinstance(it, tuple):
                text = it[0] if len(it) > 0 else None
                img = it[1] if len(it) > 1 else None
                audio = it[2] if len(it) > 2 else None
                video = it[3] if len(it) > 3 else None

                item_dict = {}
                if text is not None and str(text).strip():
                    item_dict["text"] = str(text).strip()
                if img is not None:
                    item_dict["image"] = img
                if audio is not None:
                    # Transformers audio processor expects numpy.ndarray or str path/base64
                    if isinstance(audio, tuple):
                        audio_val = audio[0]
                    else:
                        audio_val = audio

                    if hasattr(audio_val, "cpu") and hasattr(audio_val, "numpy"):
                        import numpy as np

                        audio_np = audio_val.squeeze().cpu().numpy().astype(np.float32)
                    elif hasattr(audio_val, "__array__"):
                        import numpy as np

                        audio_np = np.asarray(audio_val, dtype=np.float32).squeeze()
                    else:
                        audio_np = audio_val
                    item_dict["audio"] = audio_np
                if video is not None:
                    item_dict["video"] = video

                if not item_dict:
                    processed_inputs.append("")
                elif len(item_dict) == 1 and "text" in item_dict:
                    processed_inputs.append(item_dict["text"])
                else:
                    if "text" in item_dict:
                        if (
                            "image" in item_dict
                            and "<|image|>" not in item_dict["text"]
                        ):
                            item_dict["text"] = f"{item_dict['text']} <|image|>"
                        if (
                            "video" in item_dict
                            and "<|video|>" not in item_dict["text"]
                        ):
                            item_dict["text"] = f"{item_dict['text']} <|video|>"
                        if (
                            "audio" in item_dict
                            and "<|audio|>" not in item_dict["text"]
                        ):
                            item_dict["text"] = f"{item_dict['text']} <|audio|>"
                    processed_inputs.append(item_dict)
            elif isinstance(it, str):
                processed_inputs.append(it)
            else:
                processed_inputs.append(it)

        with self.tokenizer_lock:
            pass  # Just for consistency, tokenization happens inside encode

        with self.lock:
            target_dtype = self.dtype if self.dtype is not None else get_torch_dtype()
            device_type = "cuda" if "cuda" in self.device else "cpu"
            autocast_enabled = target_dtype is not None and (
                device_type == "cuda"
                or (device_type == "cpu" and target_dtype == torch.bfloat16)
            )
            autocast_ctx = (
                torch.autocast(
                    device_type=device_type,
                    dtype=target_dtype,
                    enabled=autocast_enabled,
                )
                if target_dtype is not None and autocast_enabled
                else nullcontext()
            )

            with autocast_ctx:
                has_media = any(
                    isinstance(x, dict)
                    and any(k in x for k in ("image", "audio", "video"))
                    for x in processed_inputs
                )

                final_prompt = None if has_media else prompt_name

                embeddings = self.model.encode(
                    processed_inputs,
                    prompt_name=final_prompt,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                )

                return embeddings.cpu().tolist()


class VisualizedBGEEmbeddingModel:
    supports_multimodal: bool = True
    supports_audio: bool = False
    supports_video: bool = False

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        weights_path: str = "Visualized_m3.pth",
        device: str = "cuda",
    ):
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.lock = threading.RLock()
        self.tokenizer_lock = threading.RLock()

        import os

        if not os.path.exists(weights_path):
            try:
                from huggingface_hub import hf_hub_download

                weights_path = hf_hub_download(
                    repo_id="BAAI/bge-visualized", filename="Visualized_m3.pth"
                )
            except Exception as e:
                logging.warning(f"Could not resolve Visualized_m3.pth from HF: {e}")

        try:
            from visual_bge.modeling import Visualized_BGE
        except ImportError:
            try:
                from .visual_bge.modeling import Visualized_BGE
            except ImportError:
                try:
                    from FlagEmbedding.visual.modeling import Visualized_BGE
                except ImportError:
                    Visualized_BGE = None

        if Visualized_BGE is None:
            raise ValueError("FlagEmbedding / visual_bge package is not installed.")

        self.model = Visualized_BGE(
            model_name_bge=model_name, model_weight=weights_path
        )
        self.model.device = torch.device(self.device)
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        return self.encode_multimodal([(t, None) for t in texts])

    def encode_multimodal(
        self, items: list[tuple[Optional[str], Optional[Image.Image]]]
    ) -> list[list[float]]:
        text_only_idx = []
        text_only_texts = []

        image_only_idx = []
        image_only_images = []

        mm_idx = []
        mm_texts = []
        mm_images = []

        results: list[list[float] | None] = [None] * len(items)

        for i, (text, image) in enumerate(items):
            if text is not None and image is not None:
                mm_idx.append(i)
                mm_texts.append(text)
                mm_images.append(image)
            elif image is not None:
                image_only_idx.append(i)
                image_only_images.append(image)
            elif text is not None:
                text_only_idx.append(i)
                text_only_texts.append(text)
            else:
                results[i] = []

        def preprocess_images(imgs):
            preprocessed = []
            for img in imgs:
                if isinstance(img, str):
                    pil_img = Image.open(img).convert("RGB")
                elif isinstance(img, Image.Image):
                    pil_img = img.convert("RGB")
                else:
                    pil_img = Image.open(img).convert("RGB")
                preprocessed.append(self.model.preprocess_val(pil_img).unsqueeze(0))
            if preprocessed:
                return torch.cat(preprocessed, dim=0)
            return None

        preprocessed_image_only = (
            preprocess_images(image_only_images) if image_only_images else None
        )
        preprocessed_mm = preprocess_images(mm_images) if mm_images else None

        text_only_tok = None
        mm_tok = None

        with self.tokenizer_lock:
            if text_only_texts:
                text_only_tok = self.model.tokenizer(
                    text_only_texts, return_tensors="pt", padding=True
                )
            if mm_texts:
                mm_tok = self.model.tokenizer(
                    mm_texts, return_tensors="pt", padding=True
                )

        with self.lock:
            with torch.no_grad():
                target_dtype = get_torch_dtype()
                device_type = "cuda" if "cuda" in self.device else "cpu"
                autocast_enabled = target_dtype is not None and (
                    device_type == "cuda"
                    or (device_type == "cpu" and target_dtype == torch.bfloat16)
                )
                autocast_ctx = (
                    torch.autocast(
                        device_type=device_type,
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    )
                    if target_dtype is not None and autocast_enabled
                    else nullcontext()
                )

                with autocast_ctx:
                    if text_only_tok is not None:
                        text_out = self.model.encode_text(text_only_tok.to(self.device))
                        text_out = text_out.cpu().tolist()
                        for i, idx in enumerate(text_only_idx):
                            results[idx] = text_out[i]

                    if preprocessed_image_only is not None:
                        img_out = self.model.encode_image(
                            preprocessed_image_only.to(self.device)
                        )
                        img_out = img_out.cpu().tolist()
                        for i, idx in enumerate(image_only_idx):
                            results[idx] = img_out[i]

                    if mm_tok is not None and preprocessed_mm is not None:
                        mm_out = self.model.encode_mm(
                            preprocessed_mm.to(self.device), mm_tok.to(self.device)
                        )
                        mm_out = mm_out.cpu().tolist()
                        for i, idx in enumerate(mm_idx):
                            results[idx] = mm_out[i]

        return [r if r is not None else [] for r in results]


# --- Model Loader (Factory) ---

_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()


def get_model(model_name: str, device: str | None = None):
    """
    Factory function to get a model instance.
    It loads real models from Hugging Face and caches them.
    """
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading model '{model_name}' on device '{device}'...")

        dtype = get_torch_dtype()
        model_kwargs = (
            {"torch_dtype": dtype} if dtype in {torch.float16, torch.bfloat16} else {}
        )

        if model_name == "google/embeddinggemma-2":
            model = EmbeddingGemma2Model(
                model_name=model_name,
                device=device,
                dtype=dtype,
            )
        elif model_name in {"bge-visualized-m3", "BAAI/bge-visualized-m3"}:
            model = VisualizedBGEEmbeddingModel(
                model_name="BAAI/bge-m3",
                weights_path="Visualized_m3.pth",
                device=device,
            )
        elif model_name in EMBEDDING_MODELS:
            if model_kwargs:
                model = SentenceTransformer(
                    model_name, device=device, model_kwargs=model_kwargs
                )
            else:
                model = SentenceTransformer(model_name, device=device)
        elif model_name in RERANK_MODELS:
            if model_kwargs:
                model = CrossEncoder(
                    model_name,
                    device=device,
                    automodel_args=model_kwargs,
                )
            else:
                model = CrossEncoder(model_name, device=device)
        elif model_name in LOGIT_GATE_MODELS:
            model = LogitGateModelWrapper(
                model_name=model_name,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        # Ensure real threading locks exist on model instance
        if not hasattr(model, "lock") or isinstance(
            getattr(model, "lock", None), MagicMock
        ):
            model.lock = threading.Lock()
        if not hasattr(model, "tokenizer_lock") or isinstance(
            getattr(model, "tokenizer_lock", None), MagicMock
        ):
            model.tokenizer_lock = threading.Lock()

        _model_cache[model_name] = model
        logging.info(f"Model '{model_name}' loaded successfully.")
        return model


def unload_model(model_name: Optional[str] = None) -> tuple[list[str], int]:
    """
    Unloads a specific model or all models from the in-memory cache.
    Reclaims CUDA VRAM (if GPU is available) and triggers garbage collection.
    Returns a tuple of (unloaded_model_names, remaining_memory_bytes).
    """
    import gc
    import psutil

    unloaded = []
    with _model_lock:
        if model_name:
            if model_name in _model_cache:
                del _model_cache[model_name]
                unloaded.append(model_name)
        else:
            unloaded = list(_model_cache.keys())
            _model_cache.clear()

    if unloaded:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        return unloaded, free_bytes
    else:
        return unloaded, psutil.virtual_memory().available
