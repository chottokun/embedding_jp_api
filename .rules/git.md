# Git Rules

## 1. Branch Strategy

* `main` への直接コミット禁止。目的別ブランチ（`feature/*`, `fix/*`, `docs/*` など）を作成すること。

## 2. Commit Policy

* 1コミット＝1目的。
* 事前に Build / Test / Lint を通過させること。
* Conventional Commits 形式に従うこと（`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `style:`）。

## 3. Pull Request

* 小さな単位で作成し、最新 `main` を取り込んでからマージすること。
* 必要に応じて Squash すること。