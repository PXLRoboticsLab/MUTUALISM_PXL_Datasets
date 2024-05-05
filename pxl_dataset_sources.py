from enum import Enum, unique

@unique
class PXL_dataset_sources(Enum):
    """
    An enumeration to define the supported datasources.
    """
    HuggingFace = "huggingface.co"
    Kaggle = "www.kaggle.com"
    Roboflow = "universe.roboflow.com"

