import os


class AuditLog:

    def write_log(self, report, filename="output/audit.log"):
        os.makedirs("output", exist_ok=True)

        with open(filename, "a") as file:
            file.write("-" * 40 + "\n")
            file.write(f"Timestamp : {report.timestamp}\n")
            file.write(f"Detected  : {report.detected}\n")
            file.write(f"Tampered  : {report.tampered}\n")
            file.write(f"Risk      : {report.risk_level}\n")
            file.write(f"Score     : {report.score}\n\n")

        print(f"Audit log updated: {filename}")
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

    logger = AuditLog()
    logger.write_log(report)