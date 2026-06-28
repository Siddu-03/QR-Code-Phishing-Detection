from datetime import datetime
from report_models import Report


class ReportGenerator:

    def generate_report(self, qr_result, tamper_result, risk_result):
        summary = (
            "Possible QR tampering detected."
            if tamper_result["tampered"]
            else "QR code appears safe."
        )

        report = Report(
            detected=qr_result["detected"],
            tampered=tamper_result["tampered"],
            confidence=tamper_result["confidence"],
            risk_level=risk_result["risk_level"],
            score=risk_result["score"],
            summary=summary,
            timestamp=datetime.now().isoformat()
        )

        return report


if __name__ == "__main__":

    qr_result = {
        "detected": True
    }

    tamper_result = {
        "tampered": True,
        "confidence": 0.82
    }

    risk_result = {
        "risk_level": "HIGH_RISK",
        "score": 87
    }

    generator = ReportGenerator()

    report = generator.generate_report(
        qr_result,
        tamper_result,
        risk_result
    )

    print(report)