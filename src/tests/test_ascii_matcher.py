from app.services.ascii_matcher import AsciiMatcher, calculate_containment


def test_extract_ascii_identifiers():
    matcher = AsciiMatcher()

    # Error codes -> extracted
    assert "0x80070035" in matcher.extract_identifiers(
        "There is an error 0x80070035 here"
    )

    # Filenames -> extracted
    tokens = matcher.extract_identifiers("Check docker-compose.yml and main.py")
    assert "docker-compose.yml" in tokens
    assert "main.py" in tokens

    # Mixed versions -> extracted
    tokens = matcher.extract_identifiers("Using win11 and v3 of the API")
    assert "win11" in tokens
    assert "v3" in tokens

    # Pure digits -> excluded
    tokens = matcher.extract_identifiers("Year 2024 with 123 errors")
    assert "2024" not in tokens
    assert "123" not in tokens

    # Stop words -> excluded
    tokens = matcher.extract_identifiers("The error what is happening")
    assert "the" not in tokens
    assert "error" not in tokens
    assert "what" not in tokens

    # Pure alphabetical identifier >= 4 -> extracted
    tokens = matcher.extract_identifiers("Testing identifier extraction")
    assert "testing" in tokens
    assert "identifier" in tokens
    assert "extraction" in tokens

    # Case insensitivity and deduplication
    tokens = matcher.extract_identifiers("V3 and v3")
    assert len([t for t in tokens if t == "v3"]) == 1

    # Trailing sentence punctuation stripping
    tokens = matcher.extract_identifiers("Check docker-compose.yml. And error 0x80070035?")
    assert "docker-compose.yml" in tokens
    assert "0x80070035" in tokens


def test_calculate_containment():
    # Short tokens (< 3 chars)
    # Exact match
    assert calculate_containment("v3", "using api v3 version", set()) == 1.0
    # Missing token
    assert calculate_containment("v3", "using api v2 version", set()) == 0.0

    # Longer tokens (>= 3 chars)
    doc_lower = "this document contains an error_code_x"
    doc_3grams = {doc_lower[i : i + 3] for i in range(len(doc_lower) - 2)}

    # Exact match
    assert calculate_containment("error_code_x", doc_lower, doc_3grams) == 1.0

    # Partial match
    # token "error_code_y" -> length 12 -> 10 3-grams
    # error_code_x -> length 12 -> 10 3-grams
    # 9 of them are shared: err, rro, ror, or_, r_c, _co, cod, ode, de_
    # token has e_y, doc has e_x
    score = calculate_containment("error_code_y", doc_lower, doc_3grams)
    assert 0.0 < score < 1.0
    assert score == 0.9  # 9/10

    # Missing token
    assert calculate_containment("missing", doc_lower, doc_3grams) == 0.0


def test_score_document():
    matcher = AsciiMatcher()
    document = "We have docker-compose.yml for v3 and error 0x80070035"

    query_tokens = ["docker-compose.yml", "v3", "0x80070035", "check"]
    # "check" is not in the document, so it will pull the score down.
    # To test exactly 1.0, we just pass the matching tokens.
    assert (
        matcher.score_document(["docker-compose.yml", "v3", "0x80070035"], document)
        == 1.0
    )

    query_tokens = ["missing_file.py"]
    assert matcher.score_document(query_tokens, document) == 0.0

    assert matcher.score_document([], document) == 0.0


def test_score_documents():
    matcher = AsciiMatcher()

    query = "v3 and 0x80070035 in docker-compose.yml"
    documents = [
        "We have docker-compose.yml for v3 and error 0x80070035",  # All match
        "Just v3 here",  # Partial match
        "Nothing matches here",  # No match
    ]

    scores = matcher.score_documents(query, documents)

    assert len(scores) == 3
    assert scores[0] == 1.0
    assert 0.0 < scores[1] < 1.0
    assert scores[2] == 0.0

    # Empty query
    scores = matcher.score_documents("what is the error", documents)  # only stop words
    assert all(s == 0.0 for s in scores)

    # Empty documents
    scores = matcher.score_documents(query, [])
    assert scores == []
