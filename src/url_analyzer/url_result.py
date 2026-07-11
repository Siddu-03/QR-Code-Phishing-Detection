"""
QR Shield - URL Result Model

This module defines the URLResult data model used to store
the complete outcome of URL analysis.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class URLResult:
    """
    Stores the result of URL analysis.
    """

    # Original URL
    url: str

    # Overall risk score (0-100)
    risk_score: int

    # Confidence of analysis (0-100%)
    confidence: float

    # Final recommendation
    # SAFE | SUSPICIOUS | HIGH_RISK
    recommendation: str

    # Reasons/flags found during analysis
    flags: List[str] = field(default_factory=list)

    # Individual analysis results
    valid_url: bool = True
    https: bool = True

    uses_ip: bool = False
    shortened: bool = False
    suspicious_tld: bool = False
    embedded_credentials: bool = False

    keywords: List[str] = field(default_factory=list)

    entropy: float = 0.0

    reputation: str = "UNKNOWN"

    # Optional message
    message: str = ""

    def is_safe(self) -> bool:
        """
        Returns True if URL is considered safe.
        """
        return self.recommendation == "SAFE"

    def is_suspicious(self) -> bool:
        """
        Returns True if URL is suspicious.
        """
        return self.recommendation == "SUSPICIOUS"

    def is_high_risk(self) -> bool:
        """
        Returns True if URL is classified as high risk.
        """
        return self.recommendation == "HIGH_RISK"

    def to_dict(self) -> dict:
        """
        Convert the object into a dictionary.
        """

        return {
            "url": self.url,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "flags": self.flags,
            "valid_url": self.valid_url,
            "https": self.https,
            "uses_ip": self.uses_ip,
            "shortened": self.shortened,
            "suspicious_tld": self.suspicious_tld,
            "embedded_credentials": self.embedded_credentials,
            "keywords": self.keywords,
            "entropy": self.entropy,
            "reputation": self.reputation,
            "message": self.message
        }

    def __str__(self):
        """
        Human-readable representation.
        """

        return (
            "\n========== URL Analysis Result ==========\n"
            f"URL                : {self.url}\n"
            f"Risk Score         : {self.risk_score}/100\n"
            f"Confidence         : {self.confidence:.2f}%\n"
            f"Recommendation     : {self.recommendation}\n"
            f"Valid URL          : {self.valid_url}\n"
            f"HTTPS              : {self.https}\n"
            f"Uses IP Address    : {self.uses_ip}\n"
            f"Shortened URL      : {self.shortened}\n"
            f"Suspicious TLD     : {self.suspicious_tld}\n"
            f"Credentials Found  : {self.embedded_credentials}\n"
            f"Keywords           : {', '.join(self.keywords) if self.keywords else 'None'}\n"
            f"Entropy            : {self.entropy:.2f}\n"
            f"Reputation         : {self.reputation}\n"
            f"Flags              : {', '.join(self.flags) if self.flags else 'None'}\n"
            f"Message            : {self.message}\n"
        )


# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------

if __name__ == "__main__":

    result = URLResult(
        url="https://paypal-login.xyz/login",
        risk_score=82,
        confidence=96.5,
        recommendation="HIGH_RISK",
        flags=[
            "Suspicious keyword detected",
            "Suspicious TLD",
            "High entropy"
        ],
        valid_url=True,
        https=True,
        uses_ip=False,
        shortened=False,
        suspicious_tld=True,
        embedded_credentials=False,
        keywords=["paypal", "login"],
        entropy=4.82,
        reputation="BLACKLISTED",
        message="Multiple phishing indicators detected."
    )

    print(result)