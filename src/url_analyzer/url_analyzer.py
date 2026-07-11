"""
QR Shield - URL Analyzer

This module combines all URL analysis components and generates
a unified URLResult for Risk Assessment.
"""

from parser import parse_url
from validators import validate_url
from domain_checks import (
    is_ip_address,
    is_shortened_url,
    has_suspicious_tld,
    has_embedded_credentials,
)
from keyword_analysis import detect_keywords
from entropy import calculate_entropy
from reputation import check_blacklist
from url_result import URLResult


class URLAnalyzer:
    """
    Main URL Analysis Engine
    """

    def __init__(self):

        self.max_score = 100

    def analyze(self, url: str) -> URLResult:
        """
        Analyze a decoded URL.

        Parameters
        ----------
        url : str

        Returns
        -------
        URLResult
        """

        score = 0
        confidence = 100.0
        flags = []

        # -----------------------------------------
        # Validate URL
        # -----------------------------------------
        if not validate_url(url):

            score += 40
            flags.append("Invalid URL syntax")

        parsed = parse_url(url)

        domain = parsed["domain"]

        # -----------------------------------------
        # HTTPS Check
        # -----------------------------------------
        if parsed["scheme"].lower() != "https":

            score += 15
            flags.append("Not using HTTPS")

        # -----------------------------------------
        # IP Address Detection
        # -----------------------------------------
        if is_ip_address(domain):

            score += 20
            flags.append("IP Address used instead of domain")

        # -----------------------------------------
        # URL Shortener
        # -----------------------------------------
        if is_shortened_url(domain):

            score += 15
            flags.append("Shortened URL detected")

        # -----------------------------------------
        # Suspicious TLD
        # -----------------------------------------
        if has_suspicious_tld(domain):

            score += 10
            flags.append("Suspicious Top-Level Domain")

        # -----------------------------------------
        # Embedded Credentials
        # -----------------------------------------
        if has_embedded_credentials(url):

            score += 20
            flags.append("Embedded credentials detected")

        # -----------------------------------------
        # Keyword Detection
        # -----------------------------------------
        keywords = detect_keywords(url)

        if keywords:

            score += min(len(keywords) * 5, 20)

            flags.append(
                f"Suspicious keywords: {', '.join(keywords)}"
            )

        # -----------------------------------------
        # Entropy
        # -----------------------------------------
        entropy = calculate_entropy(url)

        if entropy > 4.5:

            score += 15

            flags.append(
                f"High entropy ({entropy:.2f})"
            )

        # -----------------------------------------
        # Blacklist
        # -----------------------------------------
        if check_blacklist(domain):

            score += 35

            flags.append("Blacklisted domain")

        # -----------------------------------------
        # Clamp score
        # -----------------------------------------
        score = min(score, self.max_score)

        confidence = max(
            0,
            100 - (score * 0.3)
        )

        # -----------------------------------------
        # Recommendation
        # -----------------------------------------
        if score < 30:

            recommendation = "SAFE"

        elif score < 60:

            recommendation = "SUSPICIOUS"

        else:

            recommendation = "HIGH_RISK"

        # -----------------------------------------
        # Return Result
        # -----------------------------------------
        return URLResult(

            url=url,

            risk_score=score,

            confidence=round(confidence, 2),

            flags=flags,

            recommendation=recommendation
        )


# -----------------------------------------------------
# Example Usage
# -----------------------------------------------------

if __name__ == "__main__":

    analyzer = URLAnalyzer()

    sample_urls = [

        "https://google.com",

        "http://bit.ly/login",

        "http://192.168.1.10/admin",

        "https://paypal.verify-user.xyz/login",

        "https://example.com"
    ]

    for url in sample_urls:

        result = analyzer.analyze(url)

        print("=" * 60)

        print("URL:", result.url)

        print("Risk Score:", result.risk_score)

        print("Confidence:", result.confidence)

        print("Recommendation:", result.recommendation)

        print("Flags:")

        for flag in result.flags:

            print(" -", flag)