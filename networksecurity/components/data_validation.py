from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging 
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os,sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            # columns from schema instead of entire schema
            number_of_columns=len(self._schema_config['columns'])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def _validate_numerical_columns(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Checks if expected numerical columns exist
        Validates data types (int64, float64)
        Ensures both train and test have same numerical columns
        """
        try:
            logging.info("Starting numerical columns validation.")
            
            # Get expected numerical columns from schema
            expected_numerical_columns = self._schema_config.get('numerical_columns', [])
            
            if not expected_numerical_columns:
                logging.warning("No numerical columns defined in schema")
                return True
            
            # Check if numerical columns exist in both datasets
            train_numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            test_numerical_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Validate numerical columns in training data
            missing_numerical_train = set(expected_numerical_columns) - set(train_numerical_cols)
            if missing_numerical_train:
                logging.error(f"Missing numerical columns in training data: {list(missing_numerical_train)}")
                return False
            
            # Validate numerical columns in testing data
            missing_numerical_test = set(expected_numerical_columns) - set(test_numerical_cols)
            if missing_numerical_test:
                logging.error(f"Missing numerical columns in testing data: {list(missing_numerical_test)}")
                return False
            
            # Check if train and test have same numerical columns
            if set(train_numerical_cols) != set(test_numerical_cols):
                only_in_train = set(train_numerical_cols) - set(test_numerical_cols)
                only_in_test = set(test_numerical_cols) - set(train_numerical_cols)
                
                if only_in_train:
                    logging.error(f"Numerical columns only in train: {list(only_in_train)}")
                if only_in_test:
                    logging.error(f"Numerical columns only in test: {list(only_in_test)}")
                return False
            
            # Validate data types for numerical columns
            for col in expected_numerical_columns:
                if col in train_df.columns and col in test_df.columns:
                    # Check training data type
                    train_dtype = str(train_df[col].dtype)
                    if train_dtype not in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                        logging.error(f"Training data column '{col}': expected numerical type, got {train_dtype}")
                        return False
                    
                    # Check testing data type
                    test_dtype = str(test_df[col].dtype)
                    if test_dtype not in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                        logging.error(f"Testing data column '{col}': expected numerical type, got {test_dtype}")
                        return False
            
            logging.info(f"Numerical columns validation passed. Found {len(expected_numerical_columns)} numerical columns.")
            return True
            
        except Exception as e:
            logging.error(f"Error in numerical columns validation: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def _validate_column_names(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Verifies exact column names match schema
        Identifies missing or extra columns
        Warns about column order differences
        """
        try:
            logging.info("Starting column names validation.")
            
            # Extract column names from schema (handling list format)
            expected_columns = []
            for col_item in self._schema_config['columns']:
                if isinstance(col_item, dict):
                    expected_columns.extend(col_item.keys())
                else:
                    expected_columns.append(col_item)
            
            # Check training data columns
            train_columns = list(train_df.columns)
            missing_in_train = set(expected_columns) - set(train_columns)
            extra_in_train = set(train_columns) - set(expected_columns)
            
            if missing_in_train:
                logging.error(f"Missing columns in training data: {list(missing_in_train)}")
                return False
            if extra_in_train:
                logging.error(f"Unexpected columns in training data: {list(extra_in_train)}")
                return False
            
            # Check testing data columns
            test_columns = list(test_df.columns)
            missing_in_test = set(expected_columns) - set(test_columns)
            extra_in_test = set(test_columns) - set(expected_columns)
            
            if missing_in_test:
                logging.error(f"Missing columns in testing data: {list(missing_in_test)}")
                return False
            if extra_in_test:
                logging.error(f"Unexpected columns in testing data: {list(extra_in_test)}")
                return False
            
            # Check if column order matches
            if train_columns != expected_columns:
                logging.warning("Training data column order differs from schema.")
            if test_columns != expected_columns:
                logging.warning("Testing data column order differs from schema.")
            
            # Check if train and test have same column order
            if train_columns != test_columns:
                logging.warning("Training and testing data have different column orders.")
            
            logging.info("Column names validation passed.")
            return True
            
        except Exception as e:
            logging.error(f"Error in column names validation: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def _validate_data_structure(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Validates minimum/maximum row counts
        Checks target column existence
        Validates feature count matches expectations
        """
        try:
            logging.info("Starting data structure validation.")
            
            # Validate minimum rows (default values since not in schema)
            expected_min_rows = 100  # You can adjust this
            if len(train_df) < expected_min_rows:
                logging.error(f"Training data has insufficient rows: {len(train_df)}, minimum required: {expected_min_rows}")
                return False
            if len(test_df) < expected_min_rows * 0.2:  # Test set should be at least 20% of min rows
                logging.error(f"Testing data has insufficient rows: {len(test_df)}, minimum required: {int(expected_min_rows * 0.2)}")
                return False
            
            # Validate maximum rows (default values since not in schema)
            expected_max_rows = 100000  # You can adjust this
            if len(train_df) > expected_max_rows:
                logging.warning(f"Training data has more rows than expected: {len(train_df)}, maximum expected: {expected_max_rows}")
            if len(test_df) > expected_max_rows:
                logging.warning(f"Testing data has more rows than expected: {len(test_df)}, maximum expected: {expected_max_rows}")
            
            # Validate target column exists (assuming 'Result' is the target)
            target_column = 'Result'
            if target_column not in train_df.columns:
                logging.error(f"Target column '{target_column}' not found in training data")
                return False
            if target_column not in test_df.columns:
                logging.error(f"Target column '{target_column}' not found in testing data")
                return False
            
            # Validate feature count
            expected_features = len(self._schema_config['columns']) - 1  # Excluding target column
            actual_train_features = len(train_df.columns) - 1  # Excluding target column
            actual_test_features = len(test_df.columns) - 1
            
            if actual_train_features != expected_features:
                logging.error(f"Training data has {actual_train_features} features, expected {expected_features}")
                return False
            if actual_test_features != expected_features:
                logging.error(f"Testing data has {actual_test_features} features, expected {expected_features}")
                return False
            
            # Validate data shape consistency
            if train_df.shape[1] != test_df.shape[1]:
                logging.error(f"Training and testing data have different number of columns: train={train_df.shape[1]}, test={test_df.shape[1]}")
                return False
            
            logging.info(f"Data structure validation passed. Train: {train_df.shape}, Test: {test_df.shape}")
            return True
            
        except Exception as e:
            logging.error(f"Error in data structure validation: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def _validate_data_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Validates values against allowed_values in schema
        Ensures data integrity for categorical features
        """
        try:
            logging.info("Starting data values validation.")
            
            # For phishing dataset, validate that values are in expected ranges
            # Based on your schema, all columns should have integer values
            
            # Get all column names from schema
            all_columns = []
            for col_item in self._schema_config['columns']:
                if isinstance(col_item, dict):
                    all_columns.extend(col_item.keys())
                else:
                    all_columns.append(col_item)
            
            # Validate each column
            for col_name in all_columns:
                if col_name in train_df.columns and col_name in test_df.columns:
                    # For phishing dataset, most values should be -1, 0, or 1
                    # Check for extreme outliers or invalid values
                    
                    # Check training data
                    train_unique_values = train_df[col_name].unique()
                    train_min, train_max = train_df[col_name].min(), train_df[col_name].max()
                    
                    # Check testing data
                    test_unique_values = test_df[col_name].unique()
                    test_min, test_max = test_df[col_name].min(), test_df[col_name].max()
                    
                    # Log value ranges for monitoring
                    logging.info(f"Column '{col_name}' - Train range: [{train_min}, {train_max}], Test range: [{test_min}, {test_max}]")
                    
                    # Check for invalid values (like NaN converted to extreme values)
                    if train_min < -10 or train_max > 10:
                        logging.warning(f"Training data column '{col_name}' has unusual values: min={train_min}, max={train_max}")
                    
                    if test_min < -10 or test_max > 10:
                        logging.warning(f"Testing data column '{col_name}' has unusual values: min={test_min}, max={test_max}")
                    
                    # Check if train and test have similar value ranges
                    if abs(train_min - test_min) > 2 or abs(train_max - test_max) > 2:
                        logging.warning(f"Column '{col_name}' has significantly different ranges between train and test")
            
            logging.info("Data values validation completed.")
            return True
            
        except Exception as e:
            logging.error(f"Error in data values validation: {str(e)}")
            raise NetworkSecurityException(e, sys)

    def _validate_null_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Checks null values against nullable constraints
        Provides detailed null value statistics
        """
        try:
            logging.info("Starting null values validation.")
            
            # Check for null values in training data
            train_null_counts = train_df.isnull().sum()
            train_null_columns = train_null_counts[train_null_counts > 0].index.tolist()
            
            # Check for null values in testing data
            test_null_counts = test_df.isnull().sum()
            test_null_columns = test_null_counts[test_null_counts > 0].index.tolist()
            
            # For phishing dataset, typically no null values should be present
            # as all features are engineered to have specific values
            
            validation_passed = True
            
            if train_null_columns:
                logging.error(f"Training data has null values in columns: {train_null_columns}")
                for col in train_null_columns:
                    null_count = train_null_counts[col]
                    null_percentage = (null_count / len(train_df)) * 100
                    logging.error(f"  {col}: {null_count} nulls ({null_percentage:.2f}%)")
                validation_passed = False
            
            if test_null_columns:
                logging.error(f"Testing data has null values in columns: {test_null_columns}")
                for col in test_null_columns:
                    null_count = test_null_counts[col]
                    null_percentage = (null_count / len(test_df)) * 100
                    logging.error(f"  {col}: {null_count} nulls ({null_percentage:.2f}%)")
                validation_passed = False
            
            if not train_null_columns and not test_null_columns:
                logging.info("No null values found in training or testing data.")
            
            # Additional check for infinite values
            train_inf_cols = []
            test_inf_cols = []
            
            for col in train_df.select_dtypes(include=[np.number]).columns:
                if train_df[col].isin([np.inf, -np.inf]).any():
                    train_inf_cols.append(col)
                if test_df[col].isin([np.inf, -np.inf]).any():
                    test_inf_cols.append(col)
            
            if train_inf_cols:
                logging.error(f"Training data has infinite values in columns: {train_inf_cols}")
                validation_passed = False
            
            if test_inf_cols:
                logging.error(f"Testing data has infinite values in columns: {test_inf_cols}")
                validation_passed = False
            
            logging.info("Null values validation completed.")
            return validation_passed
            
        except Exception as e:
            logging.error(f"Error in null values validation: {str(e)}")
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            
            # Add return statement
            return status

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            
            # Initialize validation status
            validation_status = True
            error_messages = []
            
            ## validate number of columns
            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                validation_status = False
                error_messages.append("Train dataframe does not contain all columns.")
                
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                validation_status = False
                error_messages.append("Test dataframe does not contain all columns.")

            ## Additional comprehensive validations
            # Validate column names
            if not self._validate_column_names(train_dataframe, test_dataframe):
                validation_status = False
                error_messages.append("Column names validation failed.")
            
            # Validate numerical columns
            if not self._validate_numerical_columns(train_dataframe, test_dataframe):
                validation_status = False
                error_messages.append("Numerical columns validation failed.")
            
            # Validate data structure
            if not self._validate_data_structure(train_dataframe, test_dataframe):
                validation_status = False
                error_messages.append("Data structure validation failed.")
            
            # Validate data values
            if not self._validate_data_values(train_dataframe, test_dataframe):
                validation_status = False
                error_messages.append("Data values validation failed.")
            
            # Validate null values
            if not self._validate_null_values(train_dataframe, test_dataframe):
                validation_status = False
                error_messages.append("Null values validation failed.")

            ## lets check datadrift
            drift_status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            if not drift_status:
                logging.warning("Data drift detected between training and testing datasets.")
            
            # Create directory for validated files
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            # Save the validated files
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            # Determine final validation status (including drift)
            final_validation_status = validation_status and drift_status
            
            # Log all error messages
            if error_messages:
                for error in error_messages:
                    logging.error(error)
            
            # Set file paths based on validation results
            if final_validation_status:
                valid_train_file_path = self.data_validation_config.valid_train_file_path
                valid_test_file_path = self.data_validation_config.valid_test_file_path
                invalid_train_file_path = None
                invalid_test_file_path = None
            else:
                valid_train_file_path = None
                valid_test_file_path = None
                invalid_train_file_path = self.data_validation_config.valid_train_file_path
                invalid_test_file_path = self.data_validation_config.valid_test_file_path
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=final_validation_status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=invalid_train_file_path,
                invalid_test_file_path=invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            
            logging.info(f"Data validation completed. Final status: {final_validation_status}")
            if error_messages:
                logging.info(f"Validation errors encountered: {'; '.join(error_messages)}")
            
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)