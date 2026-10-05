#!/usr/bin/env python3
"""
Evaluation Dataset Validator & Inspector

Performs:
1. Schema and type checks (id, query, document, label, type, domain)
2. Label consistency check (label == 1 <=> type == 'positive')
3. Duplicate checks (exact query-document pairs, duplicate IDs)
4. Near-miss quality heuristics (lexical overlap vs missing key answer facts)
5. Visual sampling report generation for human inspection
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

REQUIRED_KEYS = {"id", "query", "document", "label", "type"}
VALID_TYPES = {"positive", "near_miss", "unanswerable"}

def validate_dataset(file_path: Path, verbose: bool = False) -> tuple[bool, dict]:
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return False, {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON Decode Error: {e}")
        return False, {}

    if not isinstance(data, list):
        print("❌ Error: Dataset root must be a JSON array (list).")
        return False, {}

    errors = []
    seen_ids = set()
    seen_pairs = set()
    type_counts = Counter()
    domain_counts = Counter()

    for idx, item in enumerate(data):
        # 1. Dict check
        if not isinstance(item, dict):
            errors.append(f"Item #{idx}: Expected dict, got {type(item).__name__}")
            continue

        item_id = item.get("id", f"idx_{idx}")

        # 2. Required keys
        missing_keys = REQUIRED_KEYS - set(item.keys())
        if missing_keys:
            errors.append(f"Item {item_id}: Missing keys {missing_keys}")

        # 3. Duplicate ID check
        if item_id in seen_ids:
            errors.append(f"Item {item_id}: Duplicate ID detected")
        seen_ids.add(item_id)

        # 4. Content sanity
        query = str(item.get("query", "")).strip()
        document = str(item.get("document", "")).strip()
        label = item.get("label")
        item_type = item.get("type")
        domain = item.get("domain", "unspecified")

        domain_counts[domain] += 1

        if not query:
            errors.append(f"Item {item_id}: Empty query")
        if not document:
            errors.append(f"Item {item_id}: Empty document")

        pair_key = (query, document)
        if pair_key in seen_pairs:
            errors.append(f"Item {item_id}: Exact duplicate (query, document) pair")
        seen_pairs.add(pair_key)

        # 5. Type and label consistency
        if item_type not in VALID_TYPES:
            errors.append(f"Item {item_id}: Invalid type '{item_type}'. Must be one of {VALID_TYPES}")
        else:
            type_counts[item_type] += 1
            if item_type == "positive" and label != 1:
                errors.append(f"Item {item_id}: Positive item must have label=1 (got {label})")
            elif item_type in {"near_miss", "unanswerable"} and label != 0:
                errors.append(f"Item {item_id}: {item_type} item must have label=0 (got {label})")

        # 6. Near-miss heuristics check
        if item_type == "near_miss":
            # Document length sanity
            if len(document) < 15:
                errors.append(f"Item {item_id}: Near-miss document is suspiciously short (<15 chars)")

    is_valid = len(errors) == 0

    stats = {
        "total_items": len(data),
        "type_counts": dict(type_counts),
        "domain_counts": dict(domain_counts),
        "errors_count": len(errors),
        "errors": errors[:20],  # top 20 errors
    }

    if verbose or not is_valid:
        print("\n" + "=" * 60)
        print(f"📊 DATASET VALIDATION REPORT: {file_path.name}")
        print("=" * 60)
        print(f"Total Items: {stats['total_items']}")
        print(f"Type Distribution: {stats['type_counts']}")
        print(f"Domain Distribution: {stats['domain_counts']}")
        if is_valid:
            print("✅ All integrity checks PASSED!")
        else:
            print(f"❌ {len(errors)} Errors Found:")
            for err in stats["errors"]:
                print(f"  • {err}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more errors.")
        print("=" * 60 + "\n")

    return is_valid, stats


def generate_visual_inspection_markdown(file_path: Path, output_md: Path, sample_per_group: int = 3):
    """Generates a structured Markdown report grouped by domain and type for human visual inspection."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by domain and type
    groups: dict[str, dict[str, list[dict]]] = {}
    anomalies: list[tuple[dict, str]] = []

    for item in data:
        domain = item.get("domain", "unspecified")
        item_type = item.get("type", "unknown")
        doc = item.get("document", "")
        query = item.get("query", "")

        if domain not in groups:
            groups[domain] = {t: [] for t in VALID_TYPES}
        if item_type in groups[domain]:
            groups[domain][item_type].append(item)

        # Anomaly checks for visual review
        if len(doc) < 25:
            anomalies.append((item, f"短文警告: 文書長が {len(doc)} 文字と極端に短いです"))
        elif len(doc) > 400:
            anomalies.append((item, f"長文警告: 文書長が {len(doc)} 文字と長めです"))

        if item_type == "near_miss":
            # Better Japanese keyword extraction (extract alphanumeric words, katakana blocks, kanji sequences)
            import re
            keywords = re.findall(r'[a-zA-Z0-9_\-\.]{3,}|[\u30A1-\u30FA]{3,}|[\u4E00-\u9FFF]{2,}', query)
            # Remove very generic query terms
            stop_terms = {"何ですか", "ですか", "の違い", "において", "について", "方法", "場合", "理由"}
            filtered_kws = [k for k in keywords if k not in stop_terms]
            if filtered_kws:
                match_count = sum(1 for k in filtered_kws if k in doc)
                if match_count == 0:
                    anomalies.append((item, f"キーワード不一致疑惑: クエリ主要語（{', '.join(filtered_kws[:3])}）が文書内に見当たりません"))

    lines = [
        f"# データセット目視確認・点検レポート ({file_path.name})",
        "",
        f"- **総件数**: {len(data)} 件",
        f"- **対象ドメイン数**: {len(groups)} ドメイン",
        f"- **抽出サンプル数**: 各ドメイン × 各カテゴリ 最大 {sample_per_group} 件",
        "",
        "## 目視点検の審査基準 (Inspection Criteria)",
        "1. **正解文書 (Positive)**: 質問に対する具体的かつ直接的な回答事実が文書内に明記されているか（ラベル=1の正当性）",
        "2. **紛らわしい文書 (Near-Miss)**: 質問のトピック・文脈・キーワードと高度に合致しているが、肝心の回答事実（数値・可否・対象バージョン等）が欠落しているか（ラベル=0の正当性。回答が漏れ含まれていないか）",
        "3. **無関係・回答不能 (Unanswerable)**: 質問と無関係、または全く前提が異なるか（ラベル=0の正当性）",
        "4. **専門用語・自然さ**: ドメインの専門用語（エラー名、型番、条文、成分名など）が自然か",
        "",
        "---",
        "",
    ]

    if anomalies:
        lines.append(f"## ⚠️ 自動アノマリー（要重点確認サンプル）: 計 {len(anomalies)} 件")
        lines.append("以下のサンプルは自動ヒューリスティクスにより注意フラグが立っています。特に念入りに目視確認してください：")
        lines.append("")
        for idx, (item, reason) in enumerate(anomalies[:20], 1):
            lines.append(f"### 要注意 {idx}: `[{item.get('id')}]` ({item.get('domain')} / {item.get('type')})")
            lines.append(f"- **理由**: 🔴 **{reason}**")
            lines.append(f"- **質問**: {item.get('query')}")
            lines.append(f"- **文書**:\n  > {item.get('document')}")
            lines.append(f"- **ラベル**: `{item.get('label')}`")
            lines.append("")
        if len(anomalies) > 20:
            lines.append(f"*... 他 {len(anomalies) - 20} 件のアノマリー*")
        lines.append("---")
        lines.append("")

    lines.append("## ドメイン別・カテゴリ別 目視点検サンプル")
    lines.append("")

    for domain, type_dict in sorted(groups.items()):
        lines.append(f"### ドメイン: 【{domain}】")
        for item_type in ["positive", "near_miss", "unanswerable"]:
            items = type_dict.get(item_type, [])
            sample_items = items[:sample_per_group]
            lines.append(f"#### カテゴリ: `{item_type}` (全 {len(items)} 件中 {len(sample_items)} 件抜粋)")
            lines.append("")
            for idx, item in enumerate(sample_items, 1):
                lines.append(f"##### サンプル {idx} `[{item.get('id')}]`")
                lines.append(f"- **質問 (Query)**: {item.get('query')}")
                lines.append(f"- **文書 (Document)**:\n  > {item.get('document')}")
                lines.append(f"- **正解ラベル**: `{item.get('label')}`")
                lines.append("")
        lines.append("---")
        lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 目視確認用サンプリングシートを生成しました: {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate evaluation dataset.")
    parser.add_argument("file", type=Path, help="Path to JSON dataset")
    parser.add_argument("--inspect-md", type=Path, default=None, help="Generate visual inspection Markdown sheet")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per type for inspection")
    args = parser.parse_args()

    valid, _ = validate_dataset(args.file, verbose=True)
    if valid and args.inspect_md:
        generate_visual_inspection_markdown(args.file, args.inspect_md, sample_per_group=args.samples)
    if not valid:
        sys.exit(1)
