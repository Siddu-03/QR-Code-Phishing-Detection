"""
QR Shield - Entropy Analysis

This module calculates the Shannon entropy of a URL.
Higher entropy may indicate randomly generated or
obfuscated URLs often used in phishing attacks.
"""

import math
from collections import Counter


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    Parameters
    ----------
    text : str

    Returns
    -------
    float
        Shannon entropy.
    """

    if not text:
        return 0.0

    frequencies = Counter(text)

    length = len(text)

    entropy = 0.0

    for count in frequencies.values():

        probability = count / length

        entropy -= probability * math.log2(probability)

    return round(entropy, 3)


def entropy_level(entropy: float) -> str:
    """
    Classify entropy into risk levels.
    """

    if entropy < 3.5:
        return "LOW"

    elif entropy < 4.5:
        return "MEDIUM"

    else:
        return "HIGH"


def entropy_score(entropy: float) -> int:
    """
    Convert entropy into a risk score.
    """

    if entropy < 3.5:
        return 0

    elif entropy < 4.5:
        return 10

    else:
        return 20


def analyze_entropy(url: str) -> dict:
    """
    Perform complete entropy analysis.

    Returns
    -------
    dict
    """

    entropy = calculate_entropy(url)

    return {
        "entropy": entropy,
        "level": entropy_level(entropy),
        "risk_score": entropy_score(entropy)
    }


# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------

if __name__ == "__main__":

    test_urls = [

        "https://google.com",

        "https://github.com",

        "https://example.com/login",

        "https://example.com/a8F9xP2LmQ7zYt8K8QxY73Lp",

        "https://paypal.verify-user-login.xyz/Hg82JkLmP98xY"

    ]

    for url in test_urls:

        result = analyze_entropy(url)

        print("=" * 60)
        print("URL:", url)
        print("Entropy   :", result["entropy"])
        print("Level     :", result["level"])
        print("Risk Score:", result["risk_score"])