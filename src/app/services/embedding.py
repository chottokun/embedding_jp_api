import asyncio
from typing import Any, List, Tuple, Optional
import anyio
import httpx
from PIL import Image
from fastapi import HTTPException

from .base import BaseEmbeddingService
from ..image_utils import load_image_from_source
from ..schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    Usage,
    FlatMultimodalItem,
    ContentPartText,
    ContentPartImage,
    ImageUrl,
)
from ..config import EMBEDDING_MODELS, RURI_PREFIX_MAP


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


def _apply_prefix(inputs: List[str], prefix: str) -> List[str]:
    if not prefix:
        return inputs
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in inputs]


def _normalize_raw_inputs(input_data: Any) -> list:
    if isinstance(input_data, list):
        if not input_data:
            return []
        if all(
            isinstance(x, (ContentPartText, ContentPartImage))
            or (isinstance(x, dict) and x.get("type") in {"text", "image_url"})
            for x in input_data
        ):
            return [input_data]
        return input_data
    return [input_data]


async def parse_input_item(
    item: Any, client: httpx.AsyncClient
) -> Tuple[Optional[str], Optional[Image.Image]]:
    if isinstance(item, str):
        return item, None

    if isinstance(item, FlatMultimodalItem):
        text = item.text
        img = None
        if item.image_url:
            url_val = (
                item.image_url.url
                if isinstance(item.image_url, ImageUrl)
                else item.image_url
            )
            img = await load_image_from_source(url_val, client)
        return text, img

    if isinstance(item, list):
        text_parts = []
        img = None
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
        text = "\n".join(text_parts) if text_parts else None
        return text, img

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


def _get_model_or_400(model_name: str) -> Any:
    from app.main import get_model

    if model_name not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found for embeddings.",
        )
    try:
        return get_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class EmbeddingService(BaseEmbeddingService):
    """
    Default production implementation of BaseEmbeddingService.
    """

    def __init__(self, proxy_to_tei_func: Optional[Any] = None):
        self.proxy_to_tei_func = proxy_to_tei_func

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

        has_image = any(img is not None for _, img in parsed_items)

        # TEI Proxy check (dynamically read EMBEDDING_TEI_URL from main_mod)
        tei_url = getattr(main_mod, "EMBEDDING_TEI_URL", None)
        proxy_func = getattr(main_mod, "_proxy_to_tei", self.proxy_to_tei_func)

        if tei_url and not has_image and proxy_func:
            inputs = [text for text, _ in parsed_items if text is not None]
            prefix = _determine_ruri_prefix(request)
            processed_inputs = _apply_prefix(inputs, prefix)
            data = proxy_func(
                tei_url,
                "/v1/embeddings",
                {"input": processed_inputs, "model": request.model},
            )
            return EmbeddingResponse(**data)

        model = _get_model_or_400(request.model)
        is_multimodal = getattr(model, "supports_multimodal", False) is True

        if has_image and not is_multimodal:
            raise HTTPException(
                status_code=400,
                detail=f"モデル '{request.model}' は画像入力をサポートしていません。bge-visualized-m3 などのマルチモーダル対応モデルを指定してください。",
            )

        if is_multimodal:
            prefix = _determine_ruri_prefix(request)
            processed_items = []
            for text, img in parsed_items:
                clean_text = (
                    text.strip() if isinstance(text, str) and text.strip() else None
                )
                if clean_text:
                    clean_text = _apply_prefix([clean_text], prefix)[0]
                processed_items.append((clean_text, img))

            embeddings = await anyio.to_thread.run_sync(
                model.encode_multimodal, processed_items
            )
            response_data = [
                EmbeddingData(embedding=emb, index=i)
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

        def _run_inference():
            with model.lock, model.tokenizer_lock:
                return model.encode(processed_inputs)

        vectors = await anyio.to_thread.run_sync(_run_inference)

        response_data = [
            EmbeddingData(embedding=vector, index=i)
            for i, vector in enumerate(vectors.tolist())
        ]

        return EmbeddingResponse(data=response_data, model=request.model, usage=usage)
