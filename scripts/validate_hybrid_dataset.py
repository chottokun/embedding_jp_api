#!/usr/bin/env python3
"""
Validates the hybrid evaluation dataset (benchmarks/datasets/hybrid_eval_1200.json).
Checks:
- Total records = 1,200
- ID uniqueness
- Document / query length (>= 20 characters as per requirements)
- Each query has exactly 1 positive, 1 near_miss, 1 unanswerable
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def validate_dataset(filepath: Path):
    if not filepath.exists():
        print(f"Error: Dataset file {filepath} not found.")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {filepath}")

    # 1. Total count
    if len(data) != 1200:
        print(f"Error: Expected 1200 records, got {len(data)}.")
        sys.exit(1)

    # 2. ID uniqueness
    ids = [item["id"] for item in data]
    if len(set(ids)) != 1200:
        duplicate_ids = [id for id in set(ids) if ids.count(id) > 1]
        print(f"Error: IDs are not unique. Duplicates found: {duplicate_ids[:5]}...")
        sys.exit(1)

    # 3. Content len and distribution
    query_map = defaultdict(list)
    for i, item in enumerate(data):
        q = item.get("query", "")
        doc = item.get("document", "")
        t = item.get("type", "")

        if len(q.strip()) < 20:
            print(f"Error: Query too short or empty (<20 chars) at index {i}: '{q}'")
            sys.exit(1)

        if len(doc.strip()) < 20:
            print(
                f"Error: Document too short or empty (<20 chars) at index {i}: '{doc}'"
            )
            sys.exit(1)

        query_map[q].append(t)

    # 4. Each query has exactly 1 positive, 1 near_miss, 1 unanswerable
    if len(query_map) != 400:
        print(f"Error: Expected 400 unique queries, got {len(query_map)}.")
        sys.exit(1)

    for q, types in query_map.items():
        if sorted(types) != ["near_miss", "positive", "unanswerable"]:
            print(
                "Error: Query does not have exactly [positive, near_miss, unanswerable]."
            )
            print(f"Query: {q}")
            print(f"Found types: {types}")
            sys.exit(1)

    print("All Passed: Dataset validation successful!")


if __name__ == "__main__":
    dataset_path = Path("benchmarks/datasets/hybrid_eval_1200.json")
    validate_dataset(dataset_path)
