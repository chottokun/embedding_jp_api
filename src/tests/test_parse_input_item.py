from unittest.mock import AsyncMock, patch
import httpx
import pytest

from app.schemas import (
    ContentPartAudio,
    ContentPartImage,
    ContentPartText,
    ContentPartVideo,
    FlatMultimodalItem,
    ImageUrl,
    InputAudio,
    VideoUrl,
)
from app.services.embedding import ParsedMediaItem, parse_input_item


@pytest.fixture
def mock_httpx_client():
    return AsyncMock(spec=httpx.AsyncClient)


def test_parsed_media_item_properties_and_unpacking():
    mock_img = AsyncMock()
    mock_audio = AsyncMock()
    mock_video = AsyncMock()

    item = ParsedMediaItem(
        text="test text", image=mock_img, audio=mock_audio, video=mock_video
    )

    assert item.text == "test text"
    assert item.image == mock_img
    assert item.audio == mock_audio
    assert item.video == mock_video

    # Unpack 2-element tuple for backward compatibility
    text, img = item
    assert text == "test text"
    assert img == mock_img


@pytest.mark.anyio
async def test_parse_input_item_string(mock_httpx_client):
    result = await parse_input_item("simple text input", mock_httpx_client)

    assert isinstance(result, ParsedMediaItem)
    assert result.text == "simple text input"
    assert result.image is None
    assert result.audio is None
    assert result.video is None


@pytest.mark.anyio
async def test_parse_input_item_flat_multimodal_item_text_only(mock_httpx_client):
    item = FlatMultimodalItem(text="flat text")
    result = await parse_input_item(item, mock_httpx_client)

    assert result.text == "flat text"
    assert result.image is None
    assert result.audio is None
    assert result.video is None


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_flat_multimodal_item_with_objects(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "dummy_image_obj"
    mock_load_audio.return_value = "dummy_audio_obj"
    mock_load_video.return_value = "dummy_video_obj"

    item = FlatMultimodalItem(
        text="multimodal pydantic item",
        image_url=ImageUrl(url="http://example.com/image.png"),
        input_audio=InputAudio(data="b64data", format="mp3"),
        video_url=VideoUrl(url="http://example.com/video.mp4"),
    )

    result = await parse_input_item(item, mock_httpx_client)

    assert result.text == "multimodal pydantic item"
    assert result.image == "dummy_image_obj"
    assert result.audio == "dummy_audio_obj"
    assert result.video == "dummy_video_obj"

    mock_load_image.assert_awaited_once_with(
        "http://example.com/image.png", mock_httpx_client
    )
    mock_load_audio.assert_awaited_once_with(
        "data:audio/mp3;base64,b64data", mock_httpx_client
    )
    mock_load_video.assert_awaited_once_with(
        "http://example.com/video.mp4", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_flat_multimodal_item_with_raw_strings(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "img"
    mock_load_audio.return_value = "aud"
    mock_load_video.return_value = "vid"

    item = FlatMultimodalItem(
        text="raw strings item",
        image_url="http://example.com/image_raw.png",
        audio_url="http://example.com/audio_raw.wav",
        video_url="http://example.com/video_raw.mp4",
    )

    result = await parse_input_item(item, mock_httpx_client)

    assert result.text == "raw strings item"
    assert result.image == "img"
    assert result.audio == "aud"
    assert result.video == "vid"

    mock_load_image.assert_awaited_once_with(
        "http://example.com/image_raw.png", mock_httpx_client
    )
    mock_load_audio.assert_awaited_once_with(
        "http://example.com/audio_raw.wav", mock_httpx_client
    )
    mock_load_video.assert_awaited_once_with(
        "http://example.com/video_raw.mp4", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_dict_with_nested_dicts(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "dict_img"
    mock_load_audio.return_value = "dict_aud"
    mock_load_video.return_value = "dict_vid"

    item = {
        "text": "dict text",
        "image_url": {"url": "http://example.com/dict_img.png"},
        "input_audio": {"format": "flac", "data": "audio_b64"},
        "video_url": {"url": "http://example.com/dict_vid.mp4"},
    }

    result = await parse_input_item(item, mock_httpx_client)

    assert result.text == "dict text"
    assert result.image == "dict_img"
    assert result.audio == "dict_aud"
    assert result.video == "dict_vid"

    mock_load_image.assert_awaited_once_with(
        "http://example.com/dict_img.png", mock_httpx_client
    )
    mock_load_audio.assert_awaited_once_with(
        "data:audio/flac;base64,audio_b64", mock_httpx_client
    )
    mock_load_video.assert_awaited_once_with(
        "http://example.com/dict_vid.mp4", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_dict_with_raw_strings(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "dict_str_img"
    mock_load_audio.return_value = "dict_str_aud"
    mock_load_video.return_value = "dict_str_vid"

    item = {
        "text": "dict str text",
        "image_url": "http://example.com/dict_str_img.png",
        "audio_url": "http://example.com/dict_str_aud.wav",
        "video_url": "http://example.com/dict_str_vid.mp4",
    }

    result = await parse_input_item(item, mock_httpx_client)

    assert result.text == "dict str text"
    assert result.image == "dict_str_img"
    assert result.audio == "dict_str_aud"
    assert result.video == "dict_str_vid"

    mock_load_image.assert_awaited_once_with(
        "http://example.com/dict_str_img.png", mock_httpx_client
    )
    mock_load_audio.assert_awaited_once_with(
        "http://example.com/dict_str_aud.wav", mock_httpx_client
    )
    mock_load_video.assert_awaited_once_with(
        "http://example.com/dict_str_vid.mp4", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_list_pydantic_parts(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "part_img"
    mock_load_audio.return_value = "part_aud"
    mock_load_video.return_value = "part_vid"

    items = [
        ContentPartText(type="text", text="part line 1"),
        ContentPartText(type="text", text="part line 2"),
        ContentPartImage(
            type="image_url",
            image_url=ImageUrl(url="http://example.com/part_img.jpg"),
        ),
        ContentPartAudio(
            type="input_audio",
            input_audio=InputAudio(format="ogg", data="part_audio_b64"),
        ),
        ContentPartVideo(
            type="video_url",
            video_url=VideoUrl(url="http://example.com/part_vid.webm"),
        ),
    ]

    result = await parse_input_item(items, mock_httpx_client)

    assert result.text == "part line 1\npart line 2"
    assert result.image == "part_img"
    assert result.audio == "part_aud"
    assert result.video == "part_vid"

    mock_load_image.assert_awaited_once_with(
        "http://example.com/part_img.jpg", mock_httpx_client
    )
    mock_load_audio.assert_awaited_once_with(
        "data:audio/ogg;base64,part_audio_b64", mock_httpx_client
    )
    mock_load_video.assert_awaited_once_with(
        "http://example.com/part_vid.webm", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_list_dict_parts(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "dict_part_img"
    mock_load_audio.return_value = "dict_part_aud"
    mock_load_video.return_value = "dict_part_vid"

    items = [
        {"type": "text", "text": "dict part line 1"},
        {"type": "text", "text": "dict part line 2"},
        {"type": "image_url", "image_url": "http://example.com/dict_part_img.jpg"},
        {
            "type": "input_audio",
            "input_audio": {"format": "wav", "data": "dict_part_b64"},
        },
        {"type": "video_url", "video_url": "http://example.com/dict_part_vid.mp4"},
    ]

    result = await parse_input_item(items, mock_httpx_client)

    assert result.text == "dict part line 1\ndict part line 2"
    assert result.image == "dict_part_img"
    assert result.audio == "dict_part_aud"
    assert result.video == "dict_part_vid"

    mock_load_image.assert_awaited_once_with(
        "http://example.com/dict_part_img.jpg", mock_httpx_client
    )
    mock_load_audio.assert_awaited_once_with(
        "data:audio/wav;base64,dict_part_b64", mock_httpx_client
    )
    mock_load_video.assert_awaited_once_with(
        "http://example.com/dict_part_vid.mp4", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_list_content_part_raw_and_dict_variants(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "img_var"
    mock_load_audio.return_value = "aud_var"
    mock_load_video.return_value = "vid_var"

    # Testing alternate string/dict variants inside ContentPartImage, ContentPartAudio, ContentPartVideo
    items_pydantic_str_media = [
        ContentPartImage(
            type="image_url",
            image_url="http://example.com/pydantic_str_img.png",
        ),
        ContentPartAudio(
            type="input_audio",
            input_audio="http://example.com/pydantic_str_aud.mp3",
        ),
        ContentPartVideo(
            type="video_url",
            video_url="http://example.com/pydantic_str_vid.mp4",
        ),
    ]

    result = await parse_input_item(items_pydantic_str_media, mock_httpx_client)

    assert result.text is None
    assert result.image == "img_var"
    assert result.audio == "aud_var"
    assert result.video == "vid_var"

    mock_load_image.assert_awaited_with(
        "http://example.com/pydantic_str_img.png", mock_httpx_client
    )
    mock_load_audio.assert_awaited_with(
        "http://example.com/pydantic_str_aud.mp3", mock_httpx_client
    )
    mock_load_video.assert_awaited_with(
        "http://example.com/pydantic_str_vid.mp4", mock_httpx_client
    )


@pytest.mark.anyio
@patch("app.services.embedding.load_image_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_audio_from_source", new_callable=AsyncMock)
@patch("app.services.embedding.load_video_frames_from_source", new_callable=AsyncMock)
async def test_parse_input_item_list_dict_parts_with_dict_and_video_url_obj(
    mock_load_video, mock_load_audio, mock_load_image, mock_httpx_client
):
    mock_load_image.return_value = "dict_img_obj"
    mock_load_audio.return_value = "dict_aud_str"
    mock_load_video.return_value = "dict_vid_obj"

    items = [
        {
            "type": "image_url",
            "image_url": {"url": "http://example.com/dict_img_obj.png"},
        },
        {"type": "input_audio", "input_audio": "http://example.com/dict_aud_str.wav"},
        {
            "type": "video_url",
            "video_url": VideoUrl(url="http://example.com/dict_vid_obj.mp4"),
        },
    ]

    result = await parse_input_item(items, mock_httpx_client)

    assert result.text is None
    assert result.image == "dict_img_obj"
    assert result.audio == "dict_aud_str"
    assert result.video == "dict_vid_obj"


@pytest.mark.anyio
async def test_parse_input_item_invalid_type(mock_httpx_client):
    with pytest.raises(ValueError, match="不正な入力形式です。"):
        await parse_input_item(12345, mock_httpx_client)

    with pytest.raises(ValueError, match="不正な入力形式です。"):
        await parse_input_item(object(), mock_httpx_client)
