"""
QR Shield - URL Parser

This module parses a URL into its individual components
using Python's urllib.parse library.
"""

from urllib.parse import urlparse, parse_qs
from typing import Dict, Any


def parse_url(url: str) -> Dict[str, Any]:
    """
    Parse a URL into its components.

    Parameters
    ----------
    url : str
        URL to parse.

    Returns
    -------
    dict
        Dictionary containing parsed URL components.
    """

    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "domain": parsed.hostname or "",
        "port": parsed.port,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "query_params": parse_qs(parsed.query),
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "netloc": parsed.netloc,
    }


def get_domain(url: str) -> str:
    """
    Return only the domain name from a URL.
    """

    return urlparse(url).hostname or ""


def get_path(url: str) -> str:
    """
    Return only the URL path.
    """

    return urlparse(url).path


def get_scheme(url: str) -> str:
    """
    Return URL scheme (http/https).
    """

    return urlparse(url).scheme


def has_query_parameters(url: str) -> bool:
    """
    Check whether the URL contains query parameters.
    """

    return bool(urlparse(url).query)


# -----------------------------------------------------
# Example Usage
# -----------------------------------------------------

if __name__ == "__main__":

    sample_url = (
        "https://admin:1234@example.com:8080/login"
        "?id=10&user=test#dashboard"
    )

    parsed = parse_url(sample_url)

    print("\nParsed URL Components\n")

    for key, value in parsed.items():
        print(f"{key:15}: {value}")