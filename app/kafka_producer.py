"""Kafka Producer Simulation for Real-Time IoT Data Streaming"""
import json
import time
import random
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database

class IoTDataProducer:
    """Simulates IoT sensor data streaming for the Siemens Healthineers Demo"""
    
    def __init__(self, database=None):
        self.database = database or Database()
        self.sensors = {
            'MRI-TUBE-001': ['SENSOR-MRI-01', 'SENSOR-MRI-02'],
            'CT-COIL-045': ['SENSOR-CT-01', 'SENSOR-CT-02'],
            'XRAY-DETECTOR-12': ['SENSOR-XRAY-01'],
            'ULTRASOUND-PROBE-78': ['SENSOR-ULTRA-01', 'SENSOR-ULTRA-02']
        }
        
    def generate_sensor_data(self, sku):
        sensor_id = random.choice(self.sensors.get(sku, ['SENSOR-DEFAULT-01']))
        return {
            'sku': sku,
            'sensor_id': sensor_id,
            'timestamp': datetime.now().isoformat(),
            'demand': random.randint(80, 150),
            'temperature': round(random.uniform(18.0, 25.0), 2),
            'humidity': round(random.uniform(40.0, 60.0), 2)
        }
        
    def produce_realtime_data(self, sku):
        data = self.generate_sensor_data(sku)
        # Store to database to simulate Kafka consumer behavior
        self.database.insert_realtime_data(
            sku=data['sku'],
            demand=data['demand'],
            temperature=data['temperature'],
            humidity=data['humidity'],
            sensor_id=data['sensor_id']
        )
        return data
        
    def run_indefinitely(self, interval=5):
        """Continuously stream data for the live demo"""
        print("🚀 Siemens Healthineers IoT Simulator Started...")
        while True:
            for sku in self.sensors.keys():
                self.produce_realtime_data(sku)
            time.sleep(interval)

if __name__ == "__main__":
    producer = IoTDataProducer()
    # For demo environment: run forever until container stops
    producer.run_indefinitely(interval=5)
