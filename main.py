from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig,DataValidationConfig
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
        
        logging.info("Data Validation started.")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,  
            data_validation_config=data_validation_config     
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        print("Data Validation Artifact:", data_validation_artifact)
        

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
