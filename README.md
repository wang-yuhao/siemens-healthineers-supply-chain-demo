🏥 Siemens Healthineers Supply Chain Demo

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg) ![MLOps](https://img.shields.io/badge/MLOps-MLflow-blue.svg)

**AI-Powered Demand Forecasting & Inventory Optimization for Medical Electronics Supply Chain**

> A comprehensive demonstration project showcasing end-to-end full-stack development with self-trained neural networks, automated MLOps tracking, and interactive visualization for Siemens Healthineers supply chain optimization.

## 🚀 Unified Demo Experience (New)

The project now features an integrated **Executive Dashboard** and **MLOps Admin Center**. No manual command execution is required for the demo—everything is automated through Docker.

### ⚡ Quick Start (Automated Setup)

```bash
# Clone and start all services (Dashboard, MLflow, IoT Simulator)
git clone https://github.com/wang-yuhao/siemens-healthineers-supply-chain-demo.git
cd siemens-healthineers-supply-chain-demo
docker-compose up --build
```

*   **Dashboard**: `http://localhost:8501`
*   **MLflow Registry**: `http://localhost:5000`

---

## ✨ Key Features

### 1. **Interactive Executive Dashboard**
*   **Real-time Forecasting**: Self-trained LSTM models predict SKU demand with confidence intervals.
*   **IoT Telemetry**: Live ingestion of simulated sensor data (Temperature, Humidity, Vibration).
*   **Business KPIs**: Instant visibility into stockout risks, inventory levels, and model accuracy.

### 2. **MLOps Admin Center (Integrated)**
*   **Automated Data Quality**: Run validation checks (missing values, outliers, schema) directly from the UI.
*   **Drift Monitoring**: Real-time PSI (Population Stability Index) tracking to detect model performance degradation.
*   **Experiment Registry**: Unified view of MLflow experiments and model versioning.

### 3. **Explainable AI (XAI)**
*   **SHAP & LIME**: Deep-dive into model decision-making processes for high-stakes medical logistics.
*   **Feature Importance**: Understand which global and local factors drive demand predictions.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│             Interactive Streamlit Dashboard                 │
│   (Executive View | MLOps Admin | Explainability Center)    │
└──────────────┬───────────────┬───────────────┬──────────────┘
               │               │               │
     ┌─────────▼────────┐ ┌────▼────┐ ┌────────▼────────┐
     │  Demand LSTM     │ │ MLflow  │ │  Data Quality   │
     │  (AI Engine)     │ │(Registry)│ │  (Validator)   │
     └─────────┬────────┘ └────┬────┘ └────────┬────────┘
               │               │               │
     ┌─────────▼───────────────▼───────────────▼────────┐
     │           Dockerized Microservices               │
     │      (Kafka-Sim, SQLite, Monitoring Layer)       │
     └──────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| **AI/ML** | PyTorch (LSTM), Scikit-learn, XGBoost, Optuna |
| **MLOps** | MLflow, SHAP, LIME, PSI Drift Detection |
| **Data** | Kafka (Simulated), SQLite, Pandas, NumPy |
| **Frontend** | Streamlit, Plotly (Enterprise UI Pattern) |
| **DevOps** | Docker, Docker Compose, GitHub Actions |

---

## 📧 Contact

**Yuhao Wang** - [yuhao2004@gmail.com](mailto:yuhao2004@gmail.com)
Project Link: [https://github.com/wang-yuhao/siemens-healthineers-supply-chain-demo](https://github.com/wang-yuhao/siemens-healthineers-supply-chain-demo)

*Disclaimer: This project is a demonstration for interview purposes and does not contain any proprietary Siemens Healthineers data or code.*
