class PXL_value_exception(Exception):
    """
    Custom exception class designed to indicate errors specifically related to invalid values
    in the context of the PXL dataset module.
    """
    def __init__(self, message="Given an invalid value"):
        self.message = message
        super().__init__(self.message)