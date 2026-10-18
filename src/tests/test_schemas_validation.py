import pytest
from pydantic import ValidationError

from app.schemas import (
    EmbeddingRequest,
    RerankRequest,
    DecisionsRequest,
    FlatMultimodalItem,
    ImageUrl,
    InputAudio,
    VideoUrl,
    ContentPartText,
    ContentPartImage,
    ContentPartAudio,
    ContentPartVideo,
    DecisionInputMessage,
    DecisionInputContent,
    Question,
    PredicateAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    RefusalAnswer,
    DecisionsResponse,
    UnloadRequest,
)


def test_embedding_request_dimensions_and_encoding_validation():
    req = EmbeddingRequest(
        input="test string",
        model="cl-nagoya/ruri-v3-small",
        dimensions=256,
        encoding_format="base64",
    )
    assert req.dimensions == 256
    assert req.encoding_format == "base64"

    # Dimensions < 1 should fail
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            input="test string",
            model="cl-nagoya/ruri-v3-small",
            dimensions=0,
        )

    # Invalid encoding format should fail
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            input="test string",
            model="cl-nagoya/ruri-v3-small",
            encoding_format="invalid_format",  # type: ignore
        )


def test_rerank_request_top_k_alias_and_bounds():
    req = RerankRequest.model_validate({
        "query": "query text",
        "documents": ["doc1", "doc2"],
        "model": "BAAI/bge-reranker-v2-m3",
        "top_k": 1,
    })
    assert req.top_n == 1

    # top_n negative should fail
    with pytest.raises(ValidationError):
        RerankRequest(
            query="query text",
            documents=["doc1"],
            model="BAAI/bge-reranker-v2-m3",
            top_n=-1,
        )


def test_multimodal_content_parts_schemas():
    cp_text = ContentPartText(type="text", text="sample text")
    assert cp_text.type == "text"
    assert cp_text.text == "sample text"

    cp_img = ContentPartImage(
        type="image_url",
        image_url=ImageUrl(url="data:image/jpeg;base64,1234", detail="high"),
    )
    assert cp_img.type == "image_url"
    assert isinstance(cp_img.image_url, ImageUrl)
    assert cp_img.image_url.detail == "high"

    cp_audio = ContentPartAudio(
        type="input_audio",
        input_audio=InputAudio(data="audio_b64", format="mp3"),
    )
    assert cp_audio.type == "input_audio"
    assert cp_audio.input_audio.format == "mp3"

    cp_video = ContentPartVideo(
        type="video_url",
        video_url=VideoUrl(url="http://example.com/video.mp4", sample_fps=2.0),
    )
    assert cp_video.type == "video_url"
    assert cp_video.video_url.sample_fps == 2.0


def test_flat_multimodal_item_schema():
    item = FlatMultimodalItem(
        text="hello",
        image_url="http://example.com/image.png",
        audio_url="http://example.com/audio.mp3",
    )
    assert item.text == "hello"
    assert item.image_url == "http://example.com/image.png"
    assert item.audio_url == "http://example.com/audio.mp3"


def test_decisions_request_and_answers_discriminated_union():
    msg = DecisionInputMessage(
        role="user",
        content=[DecisionInputContent(type="input_text", text="Context text")],
    )
    q = Question(
        type="predicate",
        name="is_valid",
        instructions="Is context text valid?",
    )
    req = DecisionsRequest(
        input=[msg],
        questions=[q],
    )
    assert req.model != ""
    assert len(req.input) == 1
    assert len(req.questions) == 1

    # Test Discriminated Union response parsing
    pred_ans = PredicateAnswer(name="is_valid", probability=0.9)
    choice_ans = ChoiceAnswer(
        name="pick_one", choice="A", probabilities={"A": 0.8, "B": 0.2}, confidence=0.8
    )
    score_ans = ScoreAnswer(name="score_it", score=2.5, probabilities=[0.1, 0.9])
    refusal_ans = RefusalAnswer(name="invalid_q", refusal="Cannot answer")

    resp = DecisionsResponse(
        answers=[pred_ans, choice_ans, score_ans, refusal_ans]
    )
    assert len(resp.answers) == 4
    assert resp.answers[0].type == "predicate"
    assert resp.answers[1].type == "choice"
    assert resp.answers[2].type == "score"
    assert resp.answers[3].type == "refusal"


def test_unload_request_defaults():
    req1 = UnloadRequest()
    assert req1.model is None

    req2 = UnloadRequest(model="my-model")
    assert req2.model == "my-model"
