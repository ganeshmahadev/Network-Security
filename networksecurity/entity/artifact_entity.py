from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to hold the data ingestion artifacts.
    """
    train_file_path: str
    test_file_path: str
    
@dataclass
class DataValidationArtifact:
    """
    Data Validation Artifact class to hold the data validation artifacts.
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str