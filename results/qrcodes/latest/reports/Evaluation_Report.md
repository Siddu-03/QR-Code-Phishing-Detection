# QR Shield — Evaluation Report

*Generated: 2026-07-07T14:50:54.105684+00:00*  
*Dataset: `C:\Users\SIDDU\AppData\Local\Temp\qrshield_eval_dfqriv2j`*  
*Engine version(s): n/a*

## 1. Dataset Overview

- Total images evaluated: **10000**
- Categories: **1**
- Successful pipeline runs: **10000**
- Failed pipeline runs: **0**

## 2. QR Detection Metrics

Overall — Detection Rate: **1.000**, Precision: **1.000**, Recall: **1.000**, F1: **1.000**, Accuracy: **1.000**

| Category | Detection Rate | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| uncategorized | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 10000 | 0 | 0 | 0 |

**QR Detection**

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| Actual Positive | 10000 | 0 |
| Actual Negative | 0 | 0 |

## 3. Risk / Malicious-Classification Metrics

Overall — Precision: **0.000**, Recall: **0.000**, F1: **0.000**, Accuracy: **1.000**

| Category | Detection Rate | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| uncategorized | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 10000 | 0 |

**Risk Classification**

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| Actual Positive | 0 | 0 |
| Actual Negative | 0 | 10000 |

## 4. Benchmark — Processing Time & Throughput

- Wall-clock run time: **115.12s** using **16** worker(s)
- Overall pipeline throughput: **86.87 images/sec**
- Overall average FPS (per-image): **18.40**
- Fastest image: `C:\Users\SIDDU\AppData\Local\Temp\qrshield_eval_dfqriv2j\SIIkoqkh.png` (24.54 ms)
- Slowest image: `C:\Users\SIDDU\AppData\Local\Temp\qrshield_eval_dfqriv2j\019tKMkf.png` (450.22 ms)

| Category | Images | Success Rate | Avg Time (ms) | Min (ms) | Max (ms) | Std Dev | Avg FPS |
|---|---|---|---|---|---|---|---|
| uncategorized | 10000 | 100.0% | 54.35 | 24.54 | 450.22 | 29.57 | 18.40 |

## 5. Charts

![detection_rate_per_category](plots/detection_rate_per_category.png)

![average_processing_time](plots/average_processing_time.png)

![confusion_matrix_qr_detection](plots/confusion_matrix_qr_detection.png)

![confusion_matrix_risk_classification](plots/confusion_matrix_risk_classification.png)

![risk_level_distribution](plots/risk_level_distribution.png)

![confidence_distribution](plots/confidence_distribution.png)

![processing_time_histogram](plots/processing_time_histogram.png)

![detection_success_pie](plots/detection_success_pie.png)

![false_positive_vs_negative](plots/false_positive_vs_negative.png)

## 6. Conclusions

- **QR detection** achieved an overall detection rate (recall) of **100.0%**, precision of **100.0%**, and F1 score of **1.000** across all categories.
- **Risk / malicious classification** achieved recall of **0.0%** and precision of **0.0%** (F1 = **0.000**) against the phishing/overlay-attack ground truth.
- The pipeline processed images at an average of **18.40 FPS** per worker, with an overall end-to-end throughput of **86.87 images/sec** using 16 worker(s).
- Pipeline success rate was **100.0%** (0 failure(s) out of 10000).
- The **uncategorized** category had the lowest success/robustness (100.0%), suggesting it as a priority for future preprocessing or model improvements.
