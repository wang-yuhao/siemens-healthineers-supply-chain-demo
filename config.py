"""Configuration management for enterprise integrations"""

import os
from dataclasses import dataclass

@dataclass
class SnowflakeConfig:
    account: str = os.getenv('SNOWFLAKE_ACCOUNT', '')
    user: str = os.getenv('SNOWFLAKE_USER', '')
    password: str = os.getenv('SNOWFLAKE_PASSWORD', '')
    warehouse: str = 'SUPPLY_CHAIN_WH'
    database: str = 'SIEMENS_SUPPLY_CHAIN'
    schema: str = 'PUBLIC'

@dataclass
class AppConfig:
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    export_dir: str = './exports'
    data_dir: str = './data'

snowflake_config = SnowflakeConfig()
app_config = AppConfig()
