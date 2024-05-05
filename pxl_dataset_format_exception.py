class PXL_dataset_format_exception(Exception):
    """
    A specific error for when the loader doen't support the downloaded format.
    """
    def __init__(self, message="The format of the dataset is not supported. Please create your custom loader."):
        self.message = message
        super().__init__(self.message)