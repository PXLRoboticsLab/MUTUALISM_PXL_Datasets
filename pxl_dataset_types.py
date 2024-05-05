from enum import Enum, unique
@unique
class PXL_dataset_types(Enum):
    """
    An enumeration to define the types of datasets supported in the module.
    """
    Classification = "classification"
    Segmentation = "segmentation"
    Object_Detection = "object_detection"