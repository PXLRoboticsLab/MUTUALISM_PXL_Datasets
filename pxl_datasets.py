# Import custom modules
from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_sources import PXL_dataset_sources
from .pxl_dataset_split import PXL_dataset_split
from .pxl_dataset_types import PXL_dataset_types
from .pxl_value_exception import PXL_value_exception


class PXL_datasets:
    """
    Baseclass to manage operations related to datasets,
    such as loading datasets from different sources, saving them, and displaying random images.
    """

    def __init__(self):
        """Initialize the PXL_datasets with default values."""
        self.url: str = None
        self.dataset_source: PXL_dataset_sources = None
        self.dataset_name: str = None
        self.dataset_type: PXL_dataset_types = None
        self.df_map: map = None
        self.save_directory: str = None
        self.loader: PXL_dataset_loader = None

    def load_from_url(self, loader: PXL_dataset_loader, url: str, save_directory: str):
        """
        Load dataset from a specified URL and save it to the provided directory.
        
        Parameters:
            loader (PXL_dataset_loader): The loader to use for downloading and saving the dataset.
            url (str): URL from which to download the dataset.
            save_directory (str): Directory to save the dataset.
        """
        self.loader = loader
        self.url = url
        self.save_directory = save_directory
        self._set_dataset_source_and_name_from_url(url)
        self.df_map = self.loader.download_dataset(self.url, self.dataset_source, self.dataset_name, self.save_directory)
        self.df_map = self.loader.save_dataset(self.dataset_type, self.df_map, self.save_directory)

    def load_from_save_dir(self, save_directory: str):
        """
        Abstract implementation of load dataset from a saved directory. Should be implemented in extending class. 
        
        Parameters:
            save_directory (str): Directory from which to load the dataset.
        """
        pass
    
    def load_from_df_map(self, df_map:map, save_directory:str, loader:PXL_dataset_loader=PXL_dataset_loader()):
        """
        Load dataset from a dataframemap and save it.
        
        Parameters:
            df_map (map): Dataframe map containing the dataset.
            save_directory (str): Directory to save the dataset.
            loader (PXL_dataset_loader): Loader to handle dataset operations.
        """
        self.df_map = df_map
        self.loader = loader
        self.df_map = self.loader.save_dataset(self.dataset_type, self.df_map, save_directory)


    def display_random_image(self, split:PXL_dataset_split):
        """
        Abstract implementation of display a random image. Should be implemented in extending class.
        
        Parameters:
            split (PXL_dataset_split): The split of the dataset to display an random image.
        """
        pass

    def _set_dataset_source_and_name_from_url(self, url: str):
        """
        Private method to set the dataset source and name from the given URL.
        
        Parameters:
            url (str): URL to parse for source and name.
        
        Raises:
            PXL_value_exception: If the URL is invalid or the source is unsupported.
        """
        split_url = url.split("/")
        if not len(split_url) >= 3:
            raise PXL_value_exception("Given invalid URL. Must have at least 2 '/'.")
        try:
            self.dataset_source = PXL_dataset_sources(split_url[2])
        except ValueError:
            raise PXL_value_exception("Given an url from a site that isn't supported.")
        self.dataset_name = "{}/{}".format(split_url[-2], split_url[-1])
