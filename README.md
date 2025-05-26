![image]({https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white})
# ML-Based Intrusion Detection System
## Technologies Used
### Machine Learning & Data Science
[![ML & Data Science](https://skillicons.dev/icons?i=py,numpy,pandas,tensorflow,scikitlearn,jupyter)](https://skillicons.dev)

### Development Tools
[![Development Tools](https://skillicons.dev/icons?i=bash,git,docker,vscode)](https://skillicons.dev)

## Overview
This project investigates the use of machine learning techniques against rule based models to detect cybersecurity threats using publicly available network traffic  in the publicly accesible UNSW-NB15 dataset.

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
```

## Dataset Information
- **Source**: UNSW-NB15
- **Format**: CSV
- **Description**:
  - `Protocol`: Network protocol used (e.g., TCP, UDP)
  - `Src IP`, `Dst IP`: Source and destination IP addresses (anonymized)
  - `PacketSize`: Size of the network packet
  - `Duration`: Length of the session
  - `Label`: Indicates whether the traffic is benign or an attack.

The raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of UNSW Canberra for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours. The tcpdump tool was utilised to capture 100 GB of the raw traffic (e.g., Pcap files). This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The Argus, Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label. These features are described in the UNSW-NB15_features.csv file.
## Preprocessing
- **Feature Selection**: Dropped irrelevant fields like IPs and ports.
- **Normalization**: Min-Max scaling applied to numeric features.
- **Train/Test Split**: 80% training / 20% testing.

## ML Techniques Used
- [To be filled later: Random Forest, K-Means, Decision Tree, etc.]


