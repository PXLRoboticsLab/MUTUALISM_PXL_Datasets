from roboflow import Roboflow

from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_sources import PXL_dataset_sources
from .pxl_dataset_types import PXL_dataset_types
from .pxl_value_exception import PXL_value_exception


class PXL_dataset_loader_roboflow(PXL_dataset_loader):
    """
    A loader for handling dataset dataset downloads en preprocessing for datasets hosted on Roboflow.
    """
    def download_dataset(self, url: str, dataset_source: PXL_dataset_sources, dataset_name: str, save_directory: str):
        """
        Downloads a dataset from Roboflow. Make sure to put the Roboflow_api_key.txt at the right location.
        """
        self._prepare_save_directory(save_directory)
        with open("Roboflow_api_key.txt", "r") as file:
            api_key = file.readline()
        rf = Roboflow(api_key=api_key)
        split_name = dataset_name.split("/")
        project = rf.workspace(split_name[0]).project(split_name[1])
        all_versions = project.versions()

        latest_version = all_versions[0].version
        dataset = project.version(latest_version).download("yolov5")

        ##TODO implement further from here
        
        

    def save_dataset(self, dataset_type: PXL_dataset_types, df_map: map, save_directory: str):
        """
        Save the downloaded dataset in the correct format and at a given location.
        """
        super().save_dataset(dataset_type, df_map, save_directory)

    def _prepare_save_directory(self, path: str, require_empty=True):
        super()._prepare_save_directory(path, require_empty)

    def _rename_df_map_keys_to_pxl_split_names(self, df_map: map):
        super()._rename_df_map_keys_to_pxl_split_names(df_map)

    def _save_binary_image(self, image_folder: str, image_name: str, binary_image):
        super()._save_binary_image(image_folder, image_name, binary_image)

    def _save_image_from_path(self, image_folder: str, image_name: str, image_path):
        super()._save_image_from_path(image_folder, image_name, image_path)
