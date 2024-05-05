from enum import Enum, unique

@unique
class PXL_dataset_split(Enum):
    """
    An enumeration to define the dataset split names and their beginnings to automaticly rename.
    """
    Train = "tr"
    Test = "te"
    Validation = "v"