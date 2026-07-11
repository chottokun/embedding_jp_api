from app.main import redact_pii


def test_redact_pii_no_email():
    """
    Ensure strings without any email are returned unchanged.
    """
    text = "Hello world, this is a normal string."
    assert redact_pii(text) == text


def test_redact_pii_empty_string():
    """
    Ensure empty string is returned unchanged.
    """
    assert redact_pii("") == ""


def test_redact_pii_single_email():
    """
    Ensure a single standard email is redacted correctly.
    """
    text = "Please contact us at support@example.com for more info"
    expected = "Please contact us at [REDACTED] for more info"
    assert redact_pii(text) == expected


def test_redact_pii_multiple_emails():
    """
    Ensure multiple emails in a string are all redacted.
    """
    text = "Contact alice@example.com or bob@company.org"
    expected = "Contact [REDACTED] or [REDACTED]"
    assert redact_pii(text) == expected


def test_redact_pii_complex_email():
    """
    Ensure email addresses with complex patterns (dots, plus, underscores, subdomains) are redacted.
    """
    emails = [
        "john.doe+test@subdomain.example.co.uk",
        "first_last-123@domain-name.org",
        "user@sub.domain.com",
    ]
    for email in emails:
        text = f"User email is {email}"
        expected = "User email is [REDACTED]"
        assert redact_pii(text) == expected, f"Failed to redact complex email: {email}"


def test_redact_pii_invalid_email_like():
    """
    Ensure strings that look somewhat like email addresses but are invalid are NOT redacted.
    """
    non_emails = [
        "user@",
        "@domain.com",
        "user@domain",
        "user @domain.com",
    ]
    for non_email in non_emails:
        assert redact_pii(non_email) == non_email, (
            f"Should not redact invalid email: {non_email}"
        )
