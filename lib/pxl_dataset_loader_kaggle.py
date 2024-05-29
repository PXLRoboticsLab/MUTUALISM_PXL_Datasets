from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_sources import PXL_dataset_sources
from .pxl_dataset_types import PXL_dataset_types
from .pxl_dataset_format_exception import PXL_dataset_format_exception

import shutil
from opendatasets import download as load_from_kaggle


class PXL_dataset_loader_kaggle(PXL_dataset_loader):
    """
    A loader for handling dataset dataset downloads en preprocessing for datasets hosted on Kaggle.
    """
    def download_dataset(self, url: str, dataset_source: PXL_dataset_sources, dataset_name: str, save_directory: str):
        """
        Downloads a dataset from Kaggle.
        """
        load_from_kaggle(url)
        split_name = dataset_name.split("/")
        path = self._add_single_folders_to_path(split_name[-1])
        if not self._check_and_rename_split_folders():
            raise PXL_dataset_format_exception()
        pass
        ##TODO implement further from here
        
        
    def save_dataset(self, dataset_type: PXL_dataset_types, df_map: map, save_directory: str):
        """
        Save the downloaded dataset in the correct format and at a given location.
        """
        super().save_dataset(dataset_type, df_map, save_directory)


    def _add_single_folders_to_path(self, path: str):
        return super()._add_single_folders_to_path(path)
    
    def _check_and_rename_split_folders(self, path: str):
        return super()._check_and_rename_split_folders(path)

    def _prepare_save_directory(self, path: str, require_empty=True):
        return super()._prepare_save_directory(path, require_empty)

    def _rename_df_map_keys_to_pxl_split_names(self, df_map: map):
        return super()._rename_df_map_keys_to_pxl_split_names(df_map)

    def _save_binary_image(self, image_folder: str, image_name: str, binary_image):
        return super()._save_binary_image(image_folder, image_name, binary_image)

    def _save_image_from_path(self, image_folder: str, image_name: str, image_path):
        return super()._save_image_from_path(image_folder, image_name, image_path)

