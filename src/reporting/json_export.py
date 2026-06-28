import json
import os
from dataclasses import asdict


class JsonExport:

    def save_report(self, report, filename="output/report.json"):
        os.makedirs("output", exist_ok=True)

        with open(filename, "w") as file:
            json.dump(asdict(report), file, indent=4)

        print(f"Report saved to {filename}")


if __name__ == "__main__":

    from report_generator import ReportGenerator

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

    exporter = JsonExport()
    exporter.save_report(report)