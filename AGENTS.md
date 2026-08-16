# AGENTS.md

## 1. Agent Mindset
* 私は開発者です。あなたは私のエージェントです。
* **シンプル重視:** 要求された以上のコード変更（過剰構築）は避け、最小限かつ最適な修正にとどめてください。
* **質問＝読み取り専用:** コードの解説や質問への回答時は、明示的な指示がない限りファイルを編集しないでください。

## 2. Tech Stack & Execution
* **Stack:** Python 3.12+, `uv`
* **Execution:** グローバル `python` / `pip` の使用は厳禁。**すべて `uv` 経由で実行すること。**
* **Type System:** 厳格な型ヒントを必須とする。

## 3. Mandatory Guidelines
常に以下を参照すること。
* `.rules/coding-style.md`: コーディングスタイル
* `.rules/testing.md`: テスト戦略・TDD
* `.rules/git.md`: Git 運用ルール
* `.rules/security.md`: セキュリティ要件
* `.rules/ci.md`: CI/CD要件
* `.rules/documents.md`: ドキュメント作成・運用ルール

## 4. Reference Priority
1. リポジトリ内コード -> 2. `docs/` -> 3. `README.md` -> 4. 公式ドキュメント
* 実装に合わせて `docs/` や `README.md` を常に最新化すること。

## 5. Knowledge Base & Skills
* **Knowledge Store:** ナレッジは `docs/` 配下で管理。
* **Skill Load Required:** ナレッジ調査・ドキュメント作成時は、自力で回答せず `.agents/skills/llm-wiki-docs/SKILL.md` をロードして指示に従うこと。
* **Immutable Rules:** `docs/raw/` 内のソースファイルは編集・上書き・削除を一切禁止する。
