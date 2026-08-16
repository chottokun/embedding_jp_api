## Pull Request Checks

以下を必須チェックとする。

- ruff check
- ruff format --check
- pytest
- uv audit
- gitleaks detect

## Merge Requirement

すべての CI が成功した場合のみ merge 可能。