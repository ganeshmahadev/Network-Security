import os
import sys
import json
from pymongo.mongo_client import MongoClient
import certifi
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try: 
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def cv_to_json_convertor(self, file_path: str):
        try:
            logging.info(f"Converting {file_path} to JSON format")
            data = pd.read_csv(file_path)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            logging.info(f"Inserting data into MongoDB: {self.database}.{self.collection}")
            
            self.mongo_client = MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.database= self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == "__main__":
    FILE_PATH = "network_data/phisingData.csv"
    DATABASE_NAME = "network_security"
    COLLECTION= "network_data"
    networkobj= NetworkDataExtract()
    records = networkobj.cv_to_json_convertor(FILE_PATH)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE_NAME, COLLECTION)
    logging.info(f"Number of records inserted: {no_of_records}")
    print(f"Number of records inserted: {no_of_records}")
    
    