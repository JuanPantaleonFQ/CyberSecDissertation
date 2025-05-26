# ML-Based Intrusion Detection System

## Overview
This project investigates the use of machine learning techniques to detect cybersecurity threats using publicly available network traffic datasets.

## Folder structure
![Directory Tree](https://tree.nathanfriend.io/api/v1/render?state=(%27optKs!(%27fancyI~fullPWh!false~trailingSlashI~rootDotI)~N(%27N%27.UREADME.mdUdWaset3*CSV%20Files34pcaps17-2-2025UOH*JK_H4OBalerts.csvUH*ML-DecisKTree3*Banalysis34BlogsUscXpts3*model_JorY*preprocessing_modelingY4run_BscXpt.pyUtree.txt94venvZbinZincludeZlibZpyvenv.cfg84share9916%20directoXes%2C%207%20files%27)~versK!%271%27)*%E2%94%9CG39%E2%94%82%C2%A0%C2%A0%204%E2%94%94G89QQ9%5CnBsuXcWa_G%E2%94%80%E2%94%80%20Houtput3I!trueJcomparWKionNsource!Omerged_Q%20%20U9*WatXriY.py3Z8*%01ZYXWUQONKJIHGB9843*)

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


