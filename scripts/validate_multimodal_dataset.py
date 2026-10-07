import json
import os
import sys
from PIL import Image

JSON_PATH = "benchmarks/datasets/multimodal_eval_60.json"
EXPECTED_COUNT = 60
MIN_TEXT_LENGTH = 15  # Relaxed slightly for shorter queries. The requirement says 20, let's enforce 20 for descriptions, but maybe query can be slightly shorter if needed, but wait the requirement explicitly says "meaningful length (20 chars or more) for query, positive_description, hard_negative_description". Let me enforce 20 as requested.


def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        try:
            dataset = json.load(f)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
            sys.exit(1)

    if len(dataset) != EXPECTED_COUNT:
        print(f"Error: Expected {EXPECTED_COUNT} items, found {len(dataset)}")
        sys.exit(1)

    ids = set()
    queries = set()
    pos_descs = set()
    neg_descs = set()

    errors = []

    for idx, item in enumerate(dataset):
        # Unique ID check
        item_id = item.get("id")
        if not item_id:
            errors.append(f"Item {idx} is missing an 'id'")
        elif item_id in ids:
            errors.append(f"Duplicate ID found: {item_id}")
        else:
            ids.add(item_id)

        # Image existence and validity check
        image_path = item.get("image_path")
        if not image_path:
            errors.append(f"Item {item_id} is missing 'image_path'")
        elif not os.path.exists(image_path):
            errors.append(f"Item {item_id}: Image file not found at {image_path}")
        else:
            try:
                with Image.open(image_path) as img:
                    img.verify()  # verify it is an image
                # Check dimensions (requires re-opening for size after verify)
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width <= 0 or height <= 0:
                        errors.append(
                            f"Item {item_id}: Invalid image dimensions {width}x{height}"
                        )
            except Exception as e:
                errors.append(
                    f"Item {item_id}: Failed to open/verify image at {image_path}: {e}"
                )

        # Text length and non-empty checks
        for field in ["query", "positive_description", "hard_negative_description"]:
            text = item.get(field, "")
            if not text:
                errors.append(f"Item {item_id} is missing or empty '{field}'")
            elif len(text) < 20:
                errors.append(
                    f"Item {item_id}: '{field}' is too short ({len(text)} chars, must be >= 20)"
                )

        # Uniqueness checks (Queries, positive, negative should ideally be unique across the dataset)
        query = item.get("query", "")
        if query in queries:
            errors.append(f"Item {item_id}: Duplicate query found: '{query}'")
        queries.add(query)

        pos = item.get("positive_description", "")
        if pos in pos_descs:
            errors.append(
                f"Item {item_id}: Duplicate positive_description found: '{pos}'"
            )
        pos_descs.add(pos)

        neg = item.get("hard_negative_description", "")
        if neg in neg_descs:
            errors.append(
                f"Item {item_id}: Duplicate hard_negative_description found: '{neg}'"
            )
        neg_descs.add(neg)

    if errors:
        print(f"Validation failed with {len(errors)} errors:")
        for error in errors:
            print(f" - {error}")
        sys.exit(1)

    print("All Passed: Dataset validation successful!")
    sys.exit(0)


if __name__ == "__main__":
    main()
