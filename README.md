# 🏥 Siemens Healthineers Supply Chain Demo

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://img.shields.io/badge/Python-3.10-blue.svg) [![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://img.shields.io/badge/Streamlit-1.28-red.svg) [![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](https://img.shields.io/badge/Docker-Ready-brightgreen.svg) [![MLOps](https://img.shields.io/badge/MLOps-MLflow-blue.svg)](https://img.shields.io/badge/MLOps-MLflow-blue.svg)

**AI-Powered Demand Forecasting & Inventory Optimization for Medical Electronics Supply Chain**

> A comprehensive demonstration project showcasing end-to-end full-stack development with self-trained neural networks, automated MLOps tracking, and interactive visualization for Siemens Healthineers supply chain optimization.

## 🚀 Unified Demo Experience

The project now features an integrated **Executive Dashboard**, **Global Resilience Map**, and **MLOps Admin Center**. No manual command execution is required for the demo—everything is automated through Docker.

### ⚡ Quick Start (Automated Setup)

```bash
# Clone and start all services (Dashboard, MLflow, IoT Simulator)
git clone https://github.com/wang-yuhao/siemens-healthineers-supply-chain-demo.git
cd siemens-healthineers-supply-chain-demo
docker-compose up --build
```

*   **Dashboard:** `http://localhost:8501`
*   **MLflow Registry:** `http://localhost:5000`

---

## ✨ Key Features

### 1. **Interactive Executive Dashboard**
*   **Real-time Forecasting:** Self-trained LSTM models predict SKU demand with confidence intervals.
*   **IoT Telemetry:** Live ingestion of simulated sensor data (Temperature, Humidity, Vibration).
*   **Business KPIs:** Instant visibility into stockout risks, inventory levels, and model accuracy.

### 2. **Global Supply Chain Resilience**
*   **Risk Heatmap:** Interactive world map showing status of global manufacturing sites (Erlangen, Shanghai, Malvern, etc.).
*   **Logistics Alerts:** Automated warnings for port congestion, transit delays, and cost spikes.
*   **Resilience KPIs:** Monitoring Lead Time variability and Supplier Reliability scores.

### 3. **MLOps Admin Center (Integrated)**
*   **Automated Data Quality:** Run validation checks (missing values, outliers, schema) directly from the UI.
*   **Drift Monitoring:** Real-time PSI (Population Stability Index) tracking to detect model performance degradation.
*   **Experiment Registry:** Unified view of MLflow experiments and model versioning.

### 4. **Explainable AI (XAI)**
*   **SHAP & LIME:** Deep-dive into model decision-making processes for high-stakes medical logistics.
*   **Feature Importance:** Understand which global and local factors drive demand predictions.

---

## 🏗️ Architecture

```text
[ Data Sources ] --> [ Real-time Pipeline ] --> [ Storage & Analytics ] --> [ Dashboard ]
  - IoT Sensors        - Kafka / Flink           - Snowflake (Cloud)        - Streamlit
  - ERP (SAP/Oracle)   - Feature Engineering     - SQLite (Local Cache)     - PowerBI
```

## 🛠️ Tech Stack
- **AI/ML:** PyTorch (LSTM), Scikit-Learn, Optuna, SHAP, LIME
- **MLOps:** MLflow, Prometheus, Grafana
- **Data:** Snowflake, Kafka (Simulated), Pandas, PyArrow
- **Deployment:** Docker, Docker Compose, GitHub Actions (CI/CD)

---

## 📄 Documentation
- [Technical Architecture](./TECHNICAL_ARCHITECTURE.md) - Deep dive into models and pipelines.
- [Implementation Guide](./IMPLEMENTATION_GUIDE.md) - Step-by-step enterprise integration.

---
*Created for the Senior Data Scientist interview at Siemens Healthineers.*
