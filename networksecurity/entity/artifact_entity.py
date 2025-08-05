from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to hold the data ingestion artifacts.
    """
    train_file_path: str
    test_file_path: str