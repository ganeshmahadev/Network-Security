import os
import sys
import numpy as np
import pandas as pd


"""Training Pipeline Constants"""
TARGET_COLUMN_NAME = "Result"
PIPELINE_NAME = "NetwerkSecurityPipeline"
ARTIFACTS_DIR = "Artifacts"
FILE_NAME = "phisingData.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

SCHEMA_FILE_NAME = "schema.yaml"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""Data Ingestion Constants"""
DATA_INGESTION_COLLECTION_NAME = "network_data"
DATA_INGESTION_DATABASE_NAME = "network_security"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

"""Data Validation Constants"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"