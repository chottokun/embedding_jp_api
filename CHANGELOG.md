# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multimodal (Diagram + Text) Full Support**:
  - Integrated `bge-visualized-m3` model for composite image + text and image-only embeddings in 1024 dimensions.
  - Added support for Flat schema (`FlatMultimodalItem`) and OpenAI ContentPart format (`[{"type": "text"}, {"type": "image_url"}]`).
  - Added comprehensive test and stress suite [`test_multimodal_suite.py`](test_multimodal_suite.py) testing realistic diagrams, charts, flowcharts, tables, and sketches.
  - Added Pytest integration test suite [`src/tests/test_multimodal_real.py`](src/tests/test_multimodal_real.py).
- **Air-Gap Offline Verification & Pre-Downloading**:
  - Enhanced [`src/app/download_models.py`](src/app/download_models.py) with `--verify-offline` flag for automated Hugging Face Hub offline load validation (Dry-Run).
  - Added support for downloading `Visualized_m3.pth` from `BAAI/bge-visualized`.
  - Added `.env` file loading support and hierarchical configuration priority (`OS Env` > `.env` > `config.toml` > defaults).
  - Added [`.env.example`](.env.example) template.
- **Knowledge Base (OKF v0.2)**:
  - Structured `docs/` hierarchy into `architecture/`, `domain/`, and `infrastructure/` with YAML frontmatter metadata and update log.

### Changed
- Refactored `src/app/models.py` to support flexible `device` parameters and thread-safe multimodal model inference.
- Improved schema validation in `src/app/schemas.py` to allow empty string text in multimodal requests (image-only inputs).
- Upgraded `docker-compose.yml` to support `.env` file propagation.

### Fixed
- Fixed RGBA alpha-channel transparency conversion in image pre-processing.
- Fixed SSRF guard validation to safely reject loopback, link-local, and private addresses asynchronously.
