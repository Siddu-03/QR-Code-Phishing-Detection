"""
QR Shield - Keyword Analysis

This module detects suspicious keywords commonly found
in phishing URLs.
"""

from urllib.parse import unquote

# Common phishing-related keywords
SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "verification",
    "secure",
    "account",
    "bank",
    "paypal",
    "payment",
    "wallet",
    "update",
    "confirm",
    "signin",
    "sign-in",
    "password",
    "reset",
    "gift",
    "free",
    "bonus",
    "reward",
    "win",
    "prize",
    "crypto",
    "bitcoin",
    "otp",
    "invoice",
    "urgent",
    "support",
    "security",
    "admin"
]


def detect_keywords(url: str) -> list:
    """
    Detect suspicious keywords in a URL.

    Parameters
    ----------
    url : str

    Returns
    -------
    list
        List of detected suspicious keywords.
    """

    decoded_url = unquote(url).lower()

    detected = []

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in decoded_url:
            detected.append(keyword)

    return detected


def keyword_score(keywords: list) -> int:
    """
    Calculate a risk score based on detected keywords.

    Parameters
    ----------
    keywords : list

    Returns
    -------
    int
        Risk score (0-30)
    """

    if not keywords:
        return 0

    # Maximum score capped at 30
    return min(len(keywords) * 5, 30)


def analyze_keywords(url: str) -> dict:
    """
    Perform complete keyword analysis.

    Returns
    -------
    dict
    """

    found = detect_keywords(url)

    return {
        "keywords_found": found,
        "count": len(found),
        "risk_score": keyword_score(found)
    }


# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------

if __name__ == "__main__":

    test_urls = [

        "https://google.com",

        "https://paypal.com/login",

        "https://secure-bank.xyz/verify-account",

        "https://example.com/free-gift",

        "https://bitcoin-wallet-login.com",

        "https://normalwebsite.org/about"

    ]

    for url in test_urls:

        print("=" * 60)
        print("URL:", url)

        result = analyze_keywords(url)

        print("Detected Keywords :", result["keywords_found"])
        print("Keyword Count     :", result["count"])
        print("Risk Score        :", result["risk_score"])