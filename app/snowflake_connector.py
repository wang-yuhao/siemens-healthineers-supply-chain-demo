"""Snowflake Data Warehouse Connector

This module provides integration with Snowflake cloud data warehouse
for enterprise-scale data storage and analytics.

Features:
- Cloud data platform connectivity
- ETL pipeline support
- Petabyte-scale data management
- Optimized query execution
"""

import os
import logging
from typing import Optional, Dict, List
import pandas as pd
from snowflake.connector import connect, SnowflakeConnection
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeConnector:
    """Snowflake data warehouse connector for supply chain data"""
    
    def __init__(self,  account: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 warehouse: str = 'SUPPLY_CHAIN_WH',
                 database: str = 'SIEMENS_SUPPLY_CHAIN',
                 schema: str = 'PUBLIC'):
        """Initialize Snowflake connection
        
        Args:
            account: Snowflake account identifier
            user: Username
            password: Password
            warehouse: Warehouse name
            database: Database name
            schema: Schema name
        """
        self.account = account or os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = user or os.getenv('SNOWFLAKE_USER')
        self.password = password or os.getenv('SNOWFLAKE_PASSWORD')
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.connection: Optional[SnowflakeConnection] = None
        self.engine = None
    
    def connect(self) -> bool:
        """Establish connection to Snowflake
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.connection = connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            logger.info(f"Connected to Snowflake: {self.database}.{self.schema}")
            return True
        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            return False
    
    def create_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas integration"""
        engine_url = URL(
            account=self.account,
            user=self.user,
            password=self.password,
            database=self.database,
            schema=self.schema,
            warehouse=self.warehouse
        )
        self.engine = create_engine(engine_url)
        return self.engine
    
    def upload_demand_data(self, df: pd.DataFrame, table_name: str = 'DEMAND_FORECAST') -> bool:
        """Upload demand forecast data to Snowflake
        
        Args:
            df: DataFrame with forecast data
            table_name: Target table name
            
        Returns:
            bool: True if upload successful
        """
        try:
            if self.engine is None:
                self.create_sqlalchemy_engine()
            
            df.to_sql(
                table_name.lower(),
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=10000
            )
            logger.info(f"Uploaded {len(df)} rows to {table_name}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def query(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results
        
        Args:
            sql: SQL query string
            
        Returns:
            DataFrame with query results or None
        """
        try:
            if self.connection is None:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            df = pd.DataFrame(results, columns=columns)
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
    
    def create_tables(self) -> bool:
        """Create necessary tables for supply chain data
        
        Returns:
            bool: True if tables created successfully
        """
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS DEMAND_FORECAST (
                FORECAST_ID VARCHAR(50),
                SKU_ID VARCHAR(50),
                FORECAST_DATE DATE,
                PREDICTED_DEMAND FLOAT,
                CONFIDENCE_LOWER FLOAT,
                CONFIDENCE_UPPER FLOAT,
                MODEL_VERSION VARCHAR(20),
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS INVENTORY_METRICS (
                METRIC_ID VARCHAR(50),
                SKU_ID VARCHAR(50),
                METRIC_DATE DATE,
                STOCK_LEVEL INT,
                REORDER_POINT INT,
                LEAD_TIME_DAYS INT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS MODEL_PERFORMANCE (
                MODEL_ID VARCHAR(50),
                MODEL_TYPE VARCHAR(50),
                RMSE FLOAT,
                MAE FLOAT,
                MAPE FLOAT,
                R_SQUARED FLOAT,
                TRAINING_DATE TIMESTAMP_NTZ,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
        ]
        
        try:
            if self.connection is None:
                self.connect()
            
            cursor = self.connection.cursor()
            for statement in create_statements:
                cursor.execute(statement)
            
            logger.info("Tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            return False
    
    def close(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Snowflake connection closed")


# Example usage
if __name__ == "__main__":
    # Demo mode - shows how to use the connector
    connector = SnowflakeConnector()
    
    # Note: In production, credentials would be from environment variables
    # or secure secret management system
    logger.info("Snowflake connector initialized (demo mode)")
    logger.info("To use: Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD env vars")
