"""Kafka Producer for Real-Time IoT Data Streaming"""
import json
import time
import random
from datetime import datetime
import sys
import os

# For demo purposes - simulating Kafka with simple queue
# In production, use: from kafka import KafkaProducer

class IoTDataProducer:
    """Simulates IoT sensor data streaming to Kafka"""
    
    def __init__(self, database=None):
        """Initialize IoT data producer
        
        In production environment, this would connect to Kafka:
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        """
        self.database = database
        self.sensors = {
            'MRI-TUBE-001': ['SENSOR-MRI-01', 'SENSOR-MRI-02'],
            'CT-COIL-045': ['SENSOR-CT-01', 'SENSOR-CT-02'],
            'XRAY-DETECTOR-12': ['SENSOR-XRAY-01'],
            'ULTRASOUND-PROBE-78': ['SENSOR-ULTRA-01', 'SENSOR-ULTRA-02']
        }
    
    def generate_sensor_data(self, sku):
        """Generate simulated IoT sensor data"""
        sensor_id = random.choice(self.sensors.get(sku, ['SENSOR-DEFAULT-01']))
        
        data = {
            'sku': sku,
            'sensor_id': sensor_id,
            'timestamp': datetime.now().isoformat(),
            'demand': random.randint(80, 150),
            'temperature': round(random.uniform(18.0, 25.0), 2),
            'humidity': round(random.uniform(40.0, 60.0), 2),
            'vibration': round(random.uniform(0.0, 2.0), 2),
            'power_consumption': round(random.uniform(100.0, 500.0), 2)
        }
        
        return data
    
    def produce_realtime_data(self, sku, topic='supply_chain_data'):
        """Produce and send real-time data
        
        In production:
        self.producer.send(topic, value=data)
        self.producer.flush()
        """
        data = self.generate_sensor_data(sku)
        
        # For demo: store directly to database instead of Kafka
        if self.database:
            self.database.insert_realtime_data(
                sku=data['sku'],
                demand=data['demand'],
                temperature=data['temperature'],
                humidity=data['humidity'],
                sensor_id=data['sensor_id']
            )
        
        return data
    
    def stream_continuous_data(self, duration_seconds=60, interval=2):
        """Continuously stream data for demo purposes"""
        print(f"🔴 Starting IoT data stream for {duration_seconds} seconds...")
        
        start_time = time.time()
        count = 0
        
        while (time.time() - start_time) < duration_seconds:
            # Generate data for each SKU
            for sku in self.sensors.keys():
                data = self.produce_realtime_data(sku)
                count += 1
                
                print(f"📊 [{count}] Streamed: {sku} | "
                      f"Demand: {data['demand']} | "
                      f"Temp: {data['temperature']}°C | "
                      f"Sensor: {data['sensor_id']}")
            
            time.sleep(interval)
        
        print(f"✅ Stream completed. Total records: {count}")

def main():
    """Main execution for standalone testing"""
    # For demo without database
    from database import Database
    
    db = Database()
    producer = IoTDataProducer(database=db)
    
    print("=" * 60)
    print("🚀 Siemens Healthineers - IoT Data Streaming Demo")
    print("   Simulating real-time Kafka data stream from IoT sensors")
    print("=" * 60)
    
    # Stream data for 30 seconds
    producer.stream_continuous_data(duration_seconds=30, interval=3)
    
    db.close()

if __name__ == "__main__":
    main()
