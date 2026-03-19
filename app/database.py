"""Database layer – SQLite cache and real‑time event store."""

import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class Database:
    """Persistence layer used by the dashboard and simulator."""

    def __init__(self, db_path: str = "data/supply_chain.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # check_same_thread=False is fine here because Streamlit is single‑threaded in this demo
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        self.init_sample_data()

    # ----------------------------------------------------------------- schema
    def create_tables(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_demand (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                date DATE NOT NULL,
                demand INTEGER NOT NULL,
                inventory_level INTEGER,
                lead_time_days INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS realtime_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                demand INTEGER,
                temperature REAL,
                humidity REAL,
                sensor_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                forecast_date DATE NOT NULL,
                predicted_demand REAL NOT NULL,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()

    # --------------------------------------------------------------- bootstrap
    def init_sample_data(self) -> None:
        """Populate historical_demand with synthetic data if empty."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM historical_demand")
        count = cursor.fetchone()[0]

        if count > 0:
            return

        skus = ["MRI-TUBE-001", "CT-COIL-045", "XRAY-DETECTOR-12", "ULTRASOUND-PROBE-78"]

        for sku in skus:
            base_demand = 100
            rows: list[tuple] = []
            for i in range(180):  # last 6 months
                date = (datetime.now() - timedelta(days=180 - i)).date()
                demand = int(
                    base_demand
                    + 20 * (i / 180)  # trend
                    + 15 * np.sin(2 * np.pi * i / 30)  # seasonality
                    + np.random.normal(0, 10)  # noise
                )
                demand = max(0, demand)
                rows.append((sku, date, demand, demand + 50, 3))

            cursor.executemany(
                """
                INSERT INTO historical_demand (
                    sku, date, demand, inventory_level, lead_time_days
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

        self.conn.commit()

    # --------------------------------------------------------------- queries
    def get_all_skus(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT sku FROM historical_demand ORDER BY sku")
        return [row[0] for row in cursor.fetchall()]

    def get_historical_data(self, sku: str, days: int = 180) -> pd.DataFrame:
        query = """
            SELECT date, demand, inventory_level
            FROM historical_demand
            WHERE sku = ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=(sku, days))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        return df

    def get_realtime_data(self, limit: int = 200) -> pd.DataFrame:
        """Return most recent real‑time events across all SKUs."""
        query = """
            SELECT sku, timestamp, demand, temperature, humidity, sensor_id
            FROM realtime_data
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # --------------------------------------------------------------- mutations
    def insert_realtime_data(
        self,
        sku: str,
        demand: int,
        temperature: float | None = None,
        humidity: float | None = None,
        sensor_id: str | None = None,
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO realtime_data (
                sku, timestamp, demand, temperature, humidity, sensor_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sku, datetime.now(), demand, temperature, humidity, sensor_id),
        )
        self.conn.commit()

    def save_forecast(
        self,
        sku: str,
        forecast_date,
        predicted_demand: float,
        confidence_lower: float | None = None,
        confidence_upper: float | None = None,
        model_version: str = "v1.0",
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO forecasts (
                sku,
                forecast_date,
                predicted_demand,
                confidence_interval_lower,
                confidence_interval_upper,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sku,
                forecast_date,
                predicted_demand,
                confidence_lower,
                confidence_upper,
                model_version,
            ),
        )
        self.conn.commit()

    # --------------------------------------------------------------- teardown
    def close(self) -> None:
        self.conn.close()
