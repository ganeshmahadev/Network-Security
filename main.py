from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from networksecurity.logging.logger import logging  

import sys

if __name__== "__main__":
    try:
        # Initialize Data Ingestion Configuration
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        # Create Data Ingestion Component
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Data Ingestion started.")
        # Start Data Ingestion Process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
