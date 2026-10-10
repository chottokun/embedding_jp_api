import asyncio
import base64
import math
import struct
from typing import Any, List, Tuple, Optional, Union
import anyio
import httpx
from PIL import Image
from fastapi import HTTPException

from .base import BaseEmbeddingService, get_validated_model
from ..image_utils import load_image_from_source
from ..media_utils import load_audio_from_source, load_video_frames_from_source
from ..schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    Usage,
    FlatMultimodalItem,
    ContentPartText,
    ContentPartImage,
    ContentPartAudio,
    ContentPartVideo,
    ImageUrl,
    InputAudio,
    VideoUrl,
)
from ..config import (
    EMBEDDING_MODELS,
    RURI_PREFIX_MAP,
)


class ParsedMediaItem:
    """Container for parsed text, image, audio, and video inputs."""

    def __init__(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        audio: Any = None,
        video: Any = None,
    ):
        self.text = text
        self.image = image
        self.audio = audio
        self.video = video

    def __iter__(self):
        # 2-element tuple unpack for backward compatibility with (text, img)
        return iter((self.text, self.image))


def _determine_ruri_prefix(request: EmbeddingRequest) -> str:
    prefix = ""
    if "ruri-v3" in request.model:
        if request.input_type in RURI_PREFIX_MAP:
            prefix = RURI_PREFIX_MAP[request.input_type]
        elif request.apply_ruri_prefix:
            if isinstance(request.input, str):
                prefix = RURI_PREFIX_MAP["query"]
            else:
                prefix = RURI_PREFIX_MAP["document"]
    return prefix


def _determine_gemma2_prompt(request: EmbeddingRequest) -> Optional[str]:
    # Gemma2 specific prompts
    mapping = {
        "query": "SearchQuery",
        "document": "Document",
        "classification": "Classification",
        "clustering": "Clustering",
    }
    if request.input_type in mapping:
        return mapping[request.input_type]
    return None


def _apply_prefix(inputs: List[str], prefix: str) -> List[str]:
    if not prefix:
        return inputs
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in inputs]


def _normalize_raw_inputs(input_data: Any) -> list:
    if isinstance(input_data, list):
        if not input_data:
            return []
        if all(
            isinstance(
                x,
                (
                    ContentPartText,
                    ContentPartImage,
                    ContentPartAudio,
                    ContentPartVideo,
                ),
            )
            or (
                isinstance(x, dict)
                and x.get("type") in {"text", "image_url", "input_audio", "video_url"}
            )
            for x in input_data
        ):
            return [input_data]
        return input_data
    return [input_data]


async def parse_input_item(item: Any, client: httpx.AsyncClient) -> ParsedMediaItem:
    if isinstance(item, str):
        return ParsedMediaItem(text=item)

    if isinstance(item, FlatMultimodalItem):
        text = item.text
        img = None
        audio = None
        video = None

        if item.image_url:
            url_val = (
                item.image_url.url
                if isinstance(item.image_url, ImageUrl)
                else item.image_url
            )
            img = await load_image_from_source(url_val, client)

        audio_src = item.input_audio or item.audio_url
        if audio_src:
            if isinstance(audio_src, InputAudio):
                audio_str = f"data:audio/{audio_src.format};base64,{audio_src.data}"
            else:
                audio_str = str(audio_src)
            audio = await load_audio_from_source(audio_str, client)

        if item.video_url:
            video_src = (
                item.video_url.url
                if isinstance(item.video_url, VideoUrl)
                else str(item.video_url)
            )
            video = await load_video_frames_from_source(video_src, client)

        return ParsedMediaItem(text=text, image=img, audio=audio, video=video)

    if isinstance(item, dict):
        text = item.get("text")
        img = None
        audio = None
        video = None

        if item.get("image_url"):
            val = item["image_url"]
            url_str = val.get("url") if isinstance(val, dict) else str(val)
            img = await load_image_from_source(url_str, client)

        audio_src = item.get("input_audio") or item.get("audio_url")
        if audio_src:
            if isinstance(audio_src, dict):
                audio_str = f"data:audio/{audio_src.get('format', 'wav')};base64,{audio_src.get('data', '')}"
            else:
                audio_str = str(audio_src)
            audio = await load_audio_from_source(audio_str, client)

        if item.get("video_url"):
            val = item["video_url"]
            url_str = val.get("url") if isinstance(val, dict) else str(val)
            video = await load_video_frames_from_source(url_str, client)

        return ParsedMediaItem(text=text, image=img, audio=audio, video=video)

    if isinstance(item, list):
        text_parts = []
        img = None
        audio = None
        video = None
        for part in item:
            if isinstance(part, ContentPartText) or (
                isinstance(part, dict) and part.get("type") == "text"
            ):
                text_val = (
                    part.text
                    if isinstance(part, ContentPartText)
                    else part.get("text", "")
                )
                text_parts.append(text_val)
            elif isinstance(part, ContentPartImage) or (
                isinstance(part, dict) and part.get("type") == "image_url"
            ):
                img_data_val: Any = (
                    part.image_url
                    if isinstance(part, ContentPartImage)
                    else part.get("image_url")
                )
                image_url_str: Any = (
                    img_data_val.url
                    if isinstance(img_data_val, ImageUrl)
                    else (
                        img_data_val.get("url")
                        if isinstance(img_data_val, dict)
                        else img_data_val
                    )
                )
                img = await load_image_from_source(image_url_str, client)
            elif isinstance(part, ContentPartAudio) or (
                isinstance(part, dict) and part.get("type") == "input_audio"
            ):
                audio_val = (
                    part.input_audio
                    if isinstance(part, ContentPartAudio)
                    else part.get("input_audio")
                )
                if isinstance(audio_val, InputAudio):
                    audio_str = f"data:audio/{audio_val.format};base64,{audio_val.data}"
                elif isinstance(audio_val, dict):
                    audio_str = f"data:audio/{audio_val.get('format', 'wav')};base64,{audio_val.get('data', '')}"
                else:
                    audio_str = str(audio_val)
                audio = await load_audio_from_source(audio_str, client)
            elif isinstance(part, ContentPartVideo) or (
                isinstance(part, dict) and part.get("type") == "video_url"
            ):
                video_val = (
                    part.video_url
                    if isinstance(part, ContentPartVideo)
                    else part.get("video_url")
                )
                if isinstance(video_val, VideoUrl):
                    video_str = video_val.url
                elif isinstance(video_val, dict):
                    video_str = video_val.get("url", "")
                else:
                    video_str = str(video_val)
                video = await load_video_frames_from_source(video_str, client)

        text = "\n".join(text_parts) if text_parts else None
        return ParsedMediaItem(text=text, image=img, audio=audio, video=video)

    raise ValueError("不正な入力形式です。")


def _tokenize_and_truncate_embeddings(
    model: Any, inputs: List[str]
) -> Tuple[List[str], Usage]:
    max_seq_length = getattr(model, "max_seq_length", 8192)
    if not isinstance(max_seq_length, int):
        max_seq_length = 8192
    tokenizer = model.tokenizer
    processed_inputs = list(inputs)

    with model.tokenizer_lock:
        total_tokens = 0
        special_tokens_count = tokenizer.num_special_tokens_to_add(False)
        if not isinstance(special_tokens_count, int):
            special_tokens_count = 2
        limit = max_seq_length - special_tokens_count

        batch_size = 256
        for i in range(0, len(processed_inputs), batch_size):
            batch = processed_inputs[i : i + batch_size]
            encodings = tokenizer(batch, add_special_tokens=False)

            for j, ids in enumerate(encodings["input_ids"]):
                if len(ids) > limit:
                    truncated_ids = ids[:limit]
                    truncated_text = tokenizer.decode(truncated_ids)
                    processed_inputs[i + j] = truncated_text
                    total_tokens += len(truncated_ids) + special_tokens_count
                else:
                    total_tokens += len(ids) + special_tokens_count

        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
    return processed_inputs, usage


def _format_embedding(
    vector: List[float], dimensions: Optional[int], encoding_format: str
) -> Union[List[float], str]:
    # Matryoshka dimensionality reduction
    if dimensions is not None and dimensions > 0 and dimensions < len(vector):
        vector = vector[:dimensions]
        # L2 re-normalization
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

    if encoding_format == "base64":
        # Pack list of floats as float32 little-endian binary (<f) and base64 encode
        packed = struct.pack(f"<{len(vector)}f", *vector)
        return base64.b64encode(packed).decode("utf-8")

    return vector


class EmbeddingService(BaseEmbeddingService):
    """
    Default production implementation of BaseEmbeddingService.
    """

    def __init__(
        self,
        proxy_to_tei_func: Optional[Any] = None,
        model_loader: Optional[Any] = None,
    ):
        self.proxy_to_tei_func = proxy_to_tei_func
        self.model_loader = model_loader

    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import app.main as main_mod

        if request.model not in EMBEDDING_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found for embeddings.",
            )

        raw_items = _normalize_raw_inputs(request.input)

        try:
            async with httpx.AsyncClient() as client:
                tasks = [parse_input_item(item, client) for item in raw_items]
                parsed_items: List[
                    Tuple[Optional[str], Optional[Image.Image]]
                ] = await asyncio.gather(*tasks)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        has_image = any(p.image is not None for p in parsed_items)
        has_audio = any(p.audio is not None for p in parsed_items)
        has_video = any(p.video is not None for p in parsed_items)

        # TEI Proxy check (dynamically read EMBEDDING_TEI_URL from main_mod)
        tei_url = getattr(main_mod, "EMBEDDING_TEI_URL", None)
        proxy_func = getattr(main_mod, "_proxy_to_tei", self.proxy_to_tei_func)

        if tei_url and not (has_image or has_audio or has_video) and proxy_func:
            inputs = [item.text for item in parsed_items if item.text is not None]
            prefix = _determine_ruri_prefix(request)
            processed_inputs = _apply_prefix(inputs, prefix)
            data = await proxy_func(
                tei_url,
                "/v1/embeddings",
                {"input": processed_inputs, "model": request.model},
            )
            # Apply dimensions and encoding_format post-processing to TEI response
            processed_data = []
            for item in data.get("data", []):
                raw_emb = item["embedding"]
                idx = item["index"]
                formatted_emb = _format_embedding(
                    raw_emb, request.dimensions, request.encoding_format
                )
                processed_data.append(EmbeddingData(embedding=formatted_emb, index=idx))
            data["data"] = [d.model_dump() for d in processed_data]
            return EmbeddingResponse(**data)

        model = get_validated_model(
            request.model,
            EMBEDDING_MODELS,
            "embedding",
            loader=self.model_loader,
        )

        # Guard 1: Audio capability & server flag validation
        if has_audio:
            if not getattr(model, "supports_audio", False):
                raise HTTPException(
                    status_code=400,
                    detail=f"モデル '{request.model}' は音声入力をサポートしていません。google/embeddinggemma-2 などの対応モデルを指定してください。",
                )
            import app.config as config_mod

            active_enable_audio = getattr(
                main_mod,
                "ENABLE_AUDIO_EMBEDDING",
                getattr(config_mod, "ENABLE_AUDIO_EMBEDDING", False),
            )
            if not active_enable_audio:
                raise HTTPException(
                    status_code=400,
                    detail="Audio embedding is disabled on this server. ENABLE_AUDIO_EMBEDDING=true を設定してください。",
                )

        # Guard 2: Video capability & server flag validation
        if has_video:
            if not getattr(model, "supports_video", False):
                raise HTTPException(
                    status_code=400,
                    detail=f"モデル '{request.model}' は動画入力をサポートしていません。google/embeddinggemma-2 などの対応モデルを指定してください。",
                )
            import app.config as config_mod

            active_enable_video = getattr(
                main_mod,
                "ENABLE_VIDEO_EMBEDDING",
                getattr(config_mod, "ENABLE_VIDEO_EMBEDDING", False),
            )
            if not active_enable_video:
                raise HTTPException(
                    status_code=400,
                    detail="Video embedding is disabled on this server. ENABLE_VIDEO_EMBEDDING=true を設定してください。",
                )

        # Guard 3: Image capability validation
        if has_image and not getattr(model, "supports_multimodal", False):
            raise HTTPException(
                status_code=400,
                detail=f"モデル '{request.model}' は画像入力をサポートしていません。bge-visualized-m3 などのマルチモーダル対応モデルを指定してください。",
            )

        is_multimodal = (
            has_image
            or has_audio
            or has_video
            or getattr(model, "supports_multimodal", False) is True
        )

        if is_multimodal:
            prefix = _determine_ruri_prefix(request)
            processed_items = []
            for item in parsed_items:
                clean_text = (
                    item.text.strip()
                    if isinstance(item.text, str) and item.text.strip()
                    else None
                )
                if clean_text:
                    clean_text = _apply_prefix([clean_text], prefix)[0]
                processed_items.append((clean_text, item.image, item.audio, item.video))

            # Gemma 2 explicit prompt handling
            kwargs = {}
            if "embeddinggemma-2" in request.model.lower():
                prompt = _determine_gemma2_prompt(request)
                if prompt and not (has_image or has_audio or has_video):
                    kwargs["prompt_name"] = prompt

            def run_mm_inference():
                return model.encode_multimodal(processed_items, **kwargs)

            embeddings = await anyio.to_thread.run_sync(run_mm_inference)
            response_data = [
                EmbeddingData(
                    embedding=_format_embedding(
                        emb, request.dimensions, request.encoding_format
                    ),
                    index=i,
                )
                for i, emb in enumerate(embeddings)
            ]
            usage = Usage(prompt_tokens=0, total_tokens=0)
            return EmbeddingResponse(
                data=response_data, model=request.model, usage=usage
            )

        inputs = [text if text is not None else "" for text, _ in parsed_items]
        prefix = _determine_ruri_prefix(request)
        processed_inputs = _apply_prefix(inputs, prefix)

        processed_inputs, usage = _tokenize_and_truncate_embeddings(
            model, processed_inputs
        )

        kwargs = {}
        if "embeddinggemma-2" in request.model.lower():
            prompt = _determine_gemma2_prompt(request)
            if prompt:
                kwargs["prompt_name"] = prompt

        def _run_inference():
            with model.lock, model.tokenizer_lock:
                return model.encode(processed_inputs, **kwargs)

        vectors = await anyio.to_thread.run_sync(_run_inference)

        response_data = [
            EmbeddingData(
                embedding=_format_embedding(
                    vector, request.dimensions, request.encoding_format
                ),
                index=i,
            )
            for i, vector in enumerate(vectors.tolist())
        ]

        return EmbeddingResponse(data=response_data, model=request.model, usage=usage)
