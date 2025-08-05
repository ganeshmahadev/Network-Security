import os
import sys
import pandas as pd
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import ssl
import certifi

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

#configuration of data ingestion config
from networksecurity.entity.config_entity import DataIngestionConfig
# Importing Data Ingestion Artifact
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from dotenv import load_dotenv
load_dotenv()
# MongoDB connection
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
        
    def export_collection_as_dataframe(self):
        """Retrieves data from MongoDB and converts it into a pandas DataFrame."""
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            # Fixed MongoDB client connection - removed ssl_cert_reqs parameter
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where()
            )
            collection = self.mongo_client[database_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], axis=1, inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            # Fixed: Added sys parameter
            raise NetworkSecurityException(e, sys)
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Create directory if it does not exist
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            # Export DataFrame to CSV file
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio, 
                random_state=42
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dir_path = os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(dir_path, exist_ok=True)
            # Save train and test sets to CSV files
            logging.info(f"Saving train and test sets to {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            dataframe=self.split_data_as_train_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)