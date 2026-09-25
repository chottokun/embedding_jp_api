# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **Patched Accelerate Dependency Vulnerability (CVE-2026-69112)**:
  - Upgraded `accelerate` package to `1.15.0` to resolve path traversal and denial of service vulnerabilities in checkpoint weight maps, achieving clean `uv audit` scans.

### Fixed
- **CI Gitleaks History Depth**:
  - Configured `fetch-depth: 0` in GitHub Actions checkout step to allow `gitleaks` secret detection to inspect the full commit history correctly.

### Changed
- **Code Style Alignment**:
  - Formatted all scripts in `scratch/` and `scripts/` using `ruff format` to meet repo coding standards.

### Added
- **Logit Gate & Hybrid Rerank Engine (`Qwen/Qwen2.5-1.5B-Instruct`)**:
  - Implemented dynamic dispatch on `POST /v1/rerank` using CausalLM single-forward-pass next-token logit evaluation for decisive near-miss rejection and answerability gating.
  - Added ChatML prompt template evaluating positive target tokens (`Yes,yes,はい`) and negative target tokens (`No,no,いいえ`) via LogSumExp margin $\Delta z = z_{\text{pos}} - z_{\text{neg}}$.
  - Implemented ASCII Matcher extracting alphanumeric/symbol technical identifiers and calculating 3-gram containment ratio.
  - Fused scores in logit space: $\Delta z_{\text{final}} = \Delta z + (\beta \cdot \text{Containment})$ with numerically stable sigmoid $\sigma(\Delta z_{\text{final}})$.
  - Computed normalized binary Shannon entropy $H_{\text{binary}}$ to provide model uncertainty/hesitation observability.
  - Added fail-safe sorting (`drop_failed=False` default) to prevent array mismatch crashes in upstream clients (Dify, LangChain).
  - Evaluated on expanded $N=108$ dataset on NVIDIA GeForce RTX 3060: achieved 100.0% near-miss rejection, 100.0% unanswerable rejection, 95.4% accuracy ($\tau=0.30$), 30ms/doc GPU latency, and 0.00MB VRAM leak on model unload.
- **Inference Concurrency Control with Semaphore Protection (`MAX_CONCURRENT_INFERENCES`)**:
  - Implemented `asyncio.Semaphore` limit around neural network embedding and reranking inference to prevent GPU/CPU saturation and CUDA OOM crashes.
  - Added configurable queue timeout (`INFERENCE_SEMAPHORE_TIMEOUT_SECONDS`, default 30s) returning `503 Service Unavailable` on sustained overload.
  - Added concurrency safety test in `src/tests/test_concurrency_edge.py`.
- **Client-Specific API Keys & Individual Rate Limits (`API_KEYS_MAP`)**:
  - Supported multiple API keys via `API_KEYS` environment variable (comma-separated or JSON dictionary `{key: limit_per_minute}`).
  - Applied constant-time `secrets.compare_digest` across all configured keys to prevent timing attacks.
  - Integrated per-key custom rate limit overrides into `RateLimiter` middleware.
  - Added unit test suites in `src/tests/test_auth.py` and `src/tests/test_rate_limit.py`.
- **Production Kubernetes Deployment Manifests & Documentation (`deploy/kubernetes/`)**:
  - Provided production-ready Kubernetes manifests: `deployment.yaml` (with liveness/readiness probes and Prometheus annotations), `service.yaml` (ClusterIP), and `hpa.yaml` (HorizontalPodAutoscaler scaling 1-5 pods based on CPU/Memory targets).
  - Added deployment guide in `docs/deployment.md`.
- **OpenAPI & Swagger Documentation Enhancements**:
  - Added OpenAPI tags (`Embeddings`, `Reranking`, `Models`, `Health`, `Metrics`), endpoint summaries, descriptions, and standard response codes (400, 401, 413, 429, 503).
  - Added realistic schema examples (`json_schema_extra`) for `EmbeddingRequest` and `RerankRequest`.
  - Added unit test in `src/tests/test_extended_features.py`.
- **Prometheus Metrics Instrumentation for Token Usage and Batch Size Distribution**:
  - Added `http_prompt_tokens_total` Counter labeled by model to track total token consumption.
  - Added `http_request_batch_size` Histogram labeled by endpoint with standard exponential buckets up to 256 items.
  - Added unit test suite in `src/tests/test_metrics.py`.
- **Model Warmup & Preloading on Application Startup (`PRELOAD_MODELS`)**:
  - Implemented `PRELOAD_MODELS` environment variable supporting a comma-separated list of model names to load during application lifespan initialization.
  - Eliminates first-request cold-start latency for production environments.
  - Added unit test suite in `src/tests/test_config.py` and `src/tests/test_graceful_shutdown.py`.
- **Graceful Shutdown & In-Flight Request Draining Middleware**:
  - Implemented request task tracking and graceful drain during FastAPI application lifespan shutdown (`SHUTDOWN_DRAIN_TIMEOUT_SECONDS`, default: 10s).
  - Automatically returns `503 Service Unavailable` with retry message for new incoming requests while shutting down.
  - Added unit test suite in `src/tests/test_graceful_shutdown.py`.
- **Configurable Precision & Mixed-Precision Inference (`TORCH_DTYPE`)**:
  - Added `TORCH_DTYPE` configuration supporting `float16`, `bfloat16`, and `float32`.
  - Integrated `torch.autocast` in multimodal model inference and passed `torch_dtype` to SentenceTransformer and CrossEncoder.
  - Added unit test suite in `src/tests/test_torch_dtype.py`.
- **OpenAI-Compatible `dimensions` & `encoding_format: "base64"` Support**:
  - Added Matryoshka dimension truncation with automatic L2 re-normalization.
  - Added IEEE 754 float32 little-endian Base64 embedding serialization (`encoding_format="base64"`).
  - Applied formatting consistently across local PyTorch inference, TEI proxy, and multimodal embedding flows.
  - Added unit test suite in `src/tests/test_dimensions_and_encoding.py`.
- **IP / Token-based Rate Limiter Middleware (`429 Too Many Requests`)**:
  - Implemented sliding-window token bucket rate limiter tracking requests per minute per IP/Bearer token (`RATE_LIMIT_PER_MINUTE`, default: 120).
  - Included `Retry-After` header in 429 responses and automatically exempted internal health check/metric probes (`/health`, `/healthz`, `/ready`, `/metrics`).
  - Added unit test suite in `src/tests/test_rate_limit.py`.
- **OpenAI-Compatible Models Endpoint (`GET /v1/models`)**:
  - Implemented standard OpenAI model listing endpoint returning all configured embedding and reranking models (`ModelList`, `ModelCard`).
  - Added dedicated test suite `src/tests/test_models_endpoint.py`.
- **HTTP Payload Size Limit Middleware (DoS / OOM Defense)**:
  - Enforced 32MB maximum request body size (`MAX_PAYLOAD_SIZE`), returning `413 Payload Too Large` for oversized requests before parsing.
  - Added dedicated test suite `src/tests/test_payload_limit.py`.
- **Dynamic Model Unloading & Memory Reclamation (`POST /v1/models/unload`)**:
  - Added endpoint to dynamically unload specific or all models from cache, triggering `torch.cuda.empty_cache()` and garbage collection to free RAM/VRAM.
  - Added dedicated test suite `src/tests/test_model_unload.py`.
- **Structured JSON Logging with Request ID Tracking**:
  - Integrated JSON log formatting and `X-Request-ID` correlation via ContextVars in HTTP middleware.
  - Automatically captures HTTP method, endpoint, status code, and latency in standard JSON output for APM/log aggregation.
  - Added dedicated test suite in `src/tests/test_logger.py`.
- **Prometheus Metrics Instrumentation (`/metrics`)**:
  - Integrated `prometheus_client` exposing standard Prometheus metrics for HTTP request count and latency histograms with endpoint grouping.
  - Added dedicated test suite `src/tests/test_metrics.py`.
- **Readiness Probe Endpoint (`/ready`)**:
  - Added dedicated `/ready` endpoint verifying GPU availability and loaded model cache keys, decoupling readiness from liveness (`/health`, `/healthz`).
  - Added unit test suite in `src/tests/test_ready.py`.
- **CI / CD Automated Auditing & Secret Scanning**:
  - Enforced `uv audit` and `gitleaks` in GitHub Actions CI workflow (`.github/workflows/ci.yml`) per `.rules/ci.md`.
  - Expanded `ruff check` to entire repository (`.`).

### Changed
- **Multimodal Batch Inference Optimization**:
  - Refactored `VisualizedBGEEmbeddingModel.encode_multimodal` in `src/app/models.py` to batch preprocessed image tensors and tokenized text together instead of processing items sequentially, drastically improving multi-item inference throughput.
- **Asynchronous TEI Proxy**:
  - Upgraded TEI proxy client in `src/app/main.py` to use `httpx.AsyncClient` with pooled connections, and converted `_proxy_to_tei` and service callers to full `async/await` execution to eliminate event loop blocking.

### Fixed
- **DNS Rebinding & TOCTOU Mitigation in Multimodal Image Downloads**:
  - Implemented `SafeNetworkBackend` in `src/app/image_utils.py` with custom `httpcore.AsyncNetworkBackend` that pins the TCP connection target to the validated, safe IP address resolved during SSRF validation while retaining the original Host and SNI headers.
  - Eliminated the vulnerability window between DNS resolution and HTTP stream connection.


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
