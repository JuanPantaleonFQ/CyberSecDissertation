# ML-Based Intrusion Detection System

## Overview
This project investigates the use of machine learning techniques to detect cybersecurity threats using publicly available network traffic datasets.

## Folder Structure
```text
.
├── README.md
├── dataset
│   ├── CSV Files
│   └── pcaps17-2-2025
├── merged_output
│   ├── comparation_output
│   └── merged_suricata_alerts.csv
├── output
│   ├── ML-DecisionTree
│   ├── suricata_analysis
│   └── suricata_logs
├── scripts
│   ├── model_comparator.py
│   ├── preprocessing_modeling.py
│   └── run_suricata_script.py
├── tree.txt
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share

16 directories, 7 files

## Dataset Information
- **Source**: UNSW-NB15
- **Format**: CSV
- **Description**:
  - `Protocol`: Network protocol used (e.g., TCP, UDP)
  - `Src IP`, `Dst IP`: Source and destination IP addresses (anonymized)
  - `PacketSize`: Size of the network packet
  - `Duration`: Length of the session
  - `Label`: Indicates whether the traffic is benign or an attack

## Preprocessing
- **Feature Selection**: Dropped irrelevant fields like IPs and ports.
- **Normalization**: Min-Max scaling applied to numeric features.
- **Train/Test Split**: 80% training / 20% testing.

## ML Techniques Used
- [To be filled later: Random Forest, K-Means, Decision Tree, etc.]


