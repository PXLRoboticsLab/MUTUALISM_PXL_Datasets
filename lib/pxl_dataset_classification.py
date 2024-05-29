from .pxl_datasets import PXL_datasets
from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_types import PXL_dataset_types
from .pxl_dataset_split import PXL_dataset_split

import cv2
import matplotlib.pyplot as plt

class PXL_classification_dataset(PXL_datasets):

    def __init__(self):
        """
        Initializes the classification dataset by setting the type.
        """
        super().__init__()
        self.dataset_type = PXL_dataset_types.Classification

    def load_from_url(self, loader: PXL_dataset_loader, url: str):
        """
        Loads dataset images from a specified URL using a dataset loader.
        This method delegates to the superclass method to perform the actual loading.
        """
        return super().load_from_url(loader, url)
    
    def load_from_save_dir(self, save_directory: str):
        """
        Loads the dataset from a specified save directory.
        """
        pass
    
    def load_from_df_map(self, df_map:map, save_directory:str, loader:PXL_dataset_loader=PXL_dataset_loader()):
        """
        Loads the dataset from a existing df_map, providing the option to create a new instance of an existing df_map.
        """
        return super().load_from_df_map(df_map, save_directory, loader)
    
    def print_dataset_information(self):
        """
        Print a summary of the dataset information.
        """
        super().print_dataset_information()
    
    def display_random_image(self, split:PXL_dataset_split):
        """
        Displays a random image from the dataset for a given dataset split.
        """
        df = self.df_map[split.name.lower()]
        row = df.sample(n=1).iloc[0]
        image = cv2.imread(row['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(row['image'])
        plt.show()

