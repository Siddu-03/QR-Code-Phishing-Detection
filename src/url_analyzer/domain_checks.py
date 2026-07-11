"""
QR Shield - Domain Checks

This module performs security checks on the domain portion
of a URL to identify phishing indicators.
"""

from urllib.parse import urlparse
import ipaddress


# -------------------------------
# Common URL Shorteners
# -------------------------------

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at"
}


# -------------------------------
# Suspicious Top-Level Domains
# -------------------------------

SUSPICIOUS_TLDS = {
    ".xyz",
    ".top",
    ".click",
    ".work",
    ".zip",
    ".gq",
    ".tk",
    ".ml",
    ".cf",
    ".ga"
}


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    """

    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()

    except Exception:
        return ""


def is_ip_address(domain: str) -> bool:
    """
    Check whether the domain is an IP address.
    """

    try:
        ipaddress.ip_address(domain)
        return True

    except ValueError:
        return False


def is_shortened_url(domain: str) -> bool:
    """
    Check if the domain belongs to a known URL shortener.
    """

    return domain.lower() in SHORTENERS


def has_suspicious_tld(domain: str) -> bool:
    """
    Detect suspicious top-level domains.
    """

    domain = domain.lower()

    for tld in SUSPICIOUS_TLDS:

        if domain.endswith(tld):
            return True

    return False


def has_embedded_credentials(url: str) -> bool:
    """
    Detect URLs containing username/password.

    Example:
    https://admin:123@google.com
    """

    parsed = urlparse(url)

    return parsed.username is not None or parsed.password is not None


def has_long_domain(domain: str, threshold: int = 40) -> bool:
    """
    Check whether the domain name is unusually long.
    """

    return len(domain) > threshold


def has_many_subdomains(domain: str, max_subdomains: int = 3) -> bool:
    """
    Detect excessive subdomains.

    Example:
    a.b.c.d.example.com
    """

    return domain.count(".") > max_subdomains


def analyze_domain(url: str) -> dict:
    """
    Perform complete domain analysis.
    """

    domain = extract_domain(url)

    return {
        "domain": domain,
        "uses_ip": is_ip_address(domain),
        "shortened": is_shortened_url(domain),
        "suspicious_tld": has_suspicious_tld(domain),
        "embedded_credentials": has_embedded_credentials(url),
        "long_domain": has_long_domain(domain),
        "many_subdomains": has_many_subdomains(domain),
    }


# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------

if __name__ == "__main__":

    test_urls = [

        "https://google.com",

        "http://192.168.1.10/login",

        "https://bit.ly/abcd",

        "https://admin:123@google.com",

        "https://paypal-login.xyz",

        "https://a.b.c.d.e.example.com",

        "https://averyveryveryveryveryverylongdomainnameexample.com"

    ]

    for url in test_urls:

        result = analyze_domain(url)

        print("=" * 60)
        print("URL:", url)

        for key, value in result.items():
            print(f"{key:25}: {value}")