"""
QR Shield - Reputation Checker

This module checks the reputation of a domain using
local whitelist and blacklist databases.

Future versions can integrate:
- Google Safe Browsing API
- VirusTotal API
- PhishTank API
"""

from urllib.parse import urlparse


# -------------------------------
# Known Safe Domains
# -------------------------------

WHITELIST = {
    "google.com",
    "www.google.com",
    "github.com",
    "www.github.com",
    "microsoft.com",
    "www.microsoft.com",
    "apple.com",
    "www.apple.com",
    "amazon.com",
    "www.amazon.com",
    "wikipedia.org",
    "www.wikipedia.org"
}


# -------------------------------
# Known Malicious Domains
# -------------------------------

BLACKLIST = {
    "evil.com",
    "fakebank.xyz",
    "phishing-login.top",
    "secure-update.click",
    "paypal-login.xyz",
    "login-verify.top",
    "malicious.site"
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


def check_blacklist(url: str) -> bool:
    """
    Check if domain is blacklisted.
    """

    domain = extract_domain(url)

    return domain in BLACKLIST


def check_whitelist(url: str) -> bool:
    """
    Check if domain is whitelisted.
    """

    domain = extract_domain(url)

    return domain in WHITELIST


def reputation_score(url: str) -> int:
    """
    Assign a reputation risk score.

    Returns
    -------
    int
        0 = Safe
        10 = Unknown
        40 = Blacklisted
    """

    if check_blacklist(url):
        return 40

    if check_whitelist(url):
        return 0

    return 10


def reputation_status(url: str) -> str:
    """
    Return domain reputation status.
    """

    if check_blacklist(url):
        return "BLACKLISTED"

    if check_whitelist(url):
        return "TRUSTED"

    return "UNKNOWN"


def analyze_reputation(url: str) -> dict:
    """
    Perform complete reputation analysis.
    """

    return {
        "domain": extract_domain(url),
        "status": reputation_status(url),
        "risk_score": reputation_score(url),
        "blacklisted": check_blacklist(url),
        "trusted": check_whitelist(url)
    }


# -----------------------------------------------------
# Example Usage
# -----------------------------------------------------

if __name__ == "__main__":

    test_urls = [

        "https://google.com",

        "https://github.com",

        "https://evil.com/login",

        "https://paypal-login.xyz",

        "https://unknown-example123.com"

    ]

    for url in test_urls:

        result = analyze_reputation(url)

        print("=" * 60)
        print("URL        :", url)
        print("Domain     :", result["domain"])
        print("Status     :", result["status"])
        print("Risk Score :", result["risk_score"])
        print("Trusted    :", result["trusted"])
        print("Blacklisted:", result["blacklisted"])