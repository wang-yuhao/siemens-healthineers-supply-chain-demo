"""Database Layer - SQLite/SQL integration with Kafka"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class Database:
    def __init__(self, db_path='data/supply_chain.db'):
        """Initialize database connection"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        self.init_sample_data()
    
    def create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Historical demand data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_demand (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                date DATE NOT NULL,
                demand INTEGER NOT NULL,
                inventory_level INTEGER,
                lead_time_days INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Real-time streaming data from Kafka/IoT
        cursor.execute('''
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
        ''')
        
        # Forecasting results
        cursor.execute('''
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
        ''')
        
        self.conn.commit()
    
    def init_sample_data(self):
        """Initialize with sample historical data"""
        cursor = self.conn.cursor()
        
        # Check if data exists
        cursor.execute("SELECT COUNT(*) FROM historical_demand")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Generate sample data for multiple SKUs
            skus = ['MRI-TUBE-001', 'CT-COIL-045', 'XRAY-DETECTOR-12', 'ULTRASOUND-PROBE-78']
            
            for sku in skus:
                base_demand = 100
                data = []
                
                for i in range(180):  # 6 months of data
                    date = (datetime.now() - timedelta(days=180-i)).date()
                    # Add seasonality and trend
                    demand = int(base_demand + 
                                20 * (i / 180) +  # trend
                                15 * np.sin(2 * np.pi * i / 30) +  # monthly cycle
                                np.random.normal(0, 10))  # noise
                    demand = max(0, demand)
                    
                    data.append((sku, date, demand, demand + 50, 3))
                
                cursor.executemany(
                    'INSERT INTO historical_demand (sku, date, demand, inventory_level, lead_time_days) VALUES (?, ?, ?, ?, ?)',
                    data
                )
            
            self.conn.commit()
    
    def get_all_skus(self):
        """Get list of all SKUs"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT sku FROM historical_demand")
        return [row[0] for row in cursor.fetchall()]
    
    def get_historical_data(self, sku, days=180):
        """Retrieve historical demand data"""
        query = '''
            SELECT date, demand, inventory_level 
            FROM historical_demand 
            WHERE sku = ? 
            ORDER BY date DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=(sku, days))
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    
    def get_realtime_data(self, limit=10):
        """Get recent real-time streaming data"""
        query = '''
            SELECT sku, timestamp, demand, temperature, humidity, sensor_id 
            FROM realtime_data 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def insert_realtime_data(self, sku, demand, temperature=None, humidity=None, sensor_id=None):
        """Insert real-time data from Kafka stream"""
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO realtime_data 
               (sku, timestamp, demand, temperature, humidity, sensor_id) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (sku, datetime.now(), demand, temperature, humidity, sensor_id)
        )
        self.conn.commit()
    
    def save_forecast(self, sku, forecast_date, predicted_demand, 
                     confidence_lower=None, confidence_upper=None, model_version='v1.0'):
        """Save forecast results"""
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO forecasts 
               (sku, forecast_date, predicted_demand, confidence_interval_lower, 
                confidence_interval_upper, model_version) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (sku, forecast_date, predicted_demand, confidence_lower, 
             confidence_upper, model_version)
        )
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
