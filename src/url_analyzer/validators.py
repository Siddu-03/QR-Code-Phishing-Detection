"""
QR Shield - URL Validators

This module provides validation utilities for URLs
using only Python's built-in libraries.
"""

from urllib.parse import urlparse
import ipaddress


def validate_url(url: str) -> bool:
    """
    Check whether the URL is syntactically valid.
    A valid URL must:
    - Have http or https scheme
    - Have a non-empty domain
    """

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower() in ("http", "https")
            and parsed.netloc != ""
        )

    except Exception:
        return False


def has_valid_scheme(url: str) -> bool:
    """
    Check whether the URL uses HTTP or HTTPS.
    """

    try:
        parsed = urlparse(url)

        return parsed.scheme.lower() in ("http", "https")

    except Exception:
        return False


def is_https(url: str) -> bool:
    """
    Check whether the URL uses HTTPS.
    """

    try:
        parsed = urlparse(url)

        return parsed.scheme.lower() == "https"

    except Exception:
        return False


def has_domain(url: str) -> bool:
    """
    Check whether the URL contains a valid domain.
    """

    try:
        parsed = urlparse(url)

        return parsed.hostname is not None

    except Exception:
        return False


def is_valid_ip(ip: str) -> bool:
    """
    Check if a string is a valid IP address.
    """

    try:
        ipaddress.ip_address(ip)
        return True

    except ValueError:
        return False


def is_localhost(url: str) -> bool:
    """
    Check whether URL points to localhost.
    """

    try:
        parsed = urlparse(url)

        host = parsed.hostname

        return host == "localhost"

    except Exception:
        return False


def is_private_ip(url: str) -> bool:
    """
    Check whether URL contains a private IP address.
    """

    try:

        parsed = urlparse(url)

        host = parsed.hostname

        if host is None:
            return False

        ip = ipaddress.ip_address(host)

        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
        )

    except ValueError:
        return False

    except Exception:
        return False


def validate_all(url: str) -> dict:
    """
    Perform all validation checks.
    """

    return {
        "valid_url": validate_url(url),
        "valid_scheme": has_valid_scheme(url),
        "https": is_https(url),
        "has_domain": has_domain(url),
        "localhost": is_localhost(url),
        "private_ip": is_private_ip(url),
    }


# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------

if __name__ == "__main__":

    test_urls = [

        "https://google.com",

        "http://google.com",

        "ftp://example.com",

        "http://localhost",

        "http://192.168.1.10",

        "https://example.com/login?id=123",

        "invalid-url"

    ]

    for url in test_urls:

        print("=" * 50)
        print("URL:", url)

        result = validate_all(url)

        for key, value in result.items():

            print(f"{key:15}: {value}")