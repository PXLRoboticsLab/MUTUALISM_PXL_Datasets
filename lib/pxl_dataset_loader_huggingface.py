from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_sources import PXL_dataset_sources
from .pxl_dataset_types import PXL_dataset_types
from .pxl_value_exception import PXL_value_exception

from datasets import load_dataset as load_from_huggingface


class PXL_dataset_loader_huggingface(PXL_dataset_loader):
    """
    A loader for handling dataset dataset downloads en preprocessing for datasets hosted on Huggingface.
    """
    def download_dataset(self, url: str, dataset_source: PXL_dataset_sources, dataset_name: str, save_directory: str):
        """
        Downloads a dataset from Huggingface.
        """
        self._prepare_save_directory(save_directory)
        dataset = load_from_huggingface(dataset_name)
        df_map = {key: dataset[key].to_pandas() for key in dataset.keys()}
        df_map = self._rename_df_map_keys_to_pxl_split_names(df_map)
        return df_map
        
        

    def save_dataset(self, dataset_type: PXL_dataset_types, df_map: map, save_directory: str):
        """
        Save the downloaded dataset in the correct format and at a given location.
        """
        for key in df_map:
            image_counter = 0
            for index, row in df_map.get(key).iterrows():
                if dataset_type == PXL_dataset_types.Classification:
                    try:
                        image_folder = "{}{}/{}/".format(save_directory, key.name, str(row[label_column_name]))
                    except KeyError:
                        raise PXL_value_exception("The given label_column_name is not valid. The options are {}".format(row.keys()))
                else:
                    image_folder = "{}{}/".format(save_directory, key.name)
                self._prepare_save_directory(save_directory, require_empty=False)
                binary_image = row['image'].get('bytes', None)
                image_path = row['image'].get('path', None)
                image_name = "{}.png".format(image_counter)
                if binary_image:
                    self._save_binary_image(image_folder, image_name, binary_image)
                elif image_path:
                    self._save_image_from_path(image_folder, image_name, image_path)
                image_counter += 1

    def _prepare_save_directory(self, path: str, require_empty=True):
        super()._prepare_save_directory(path, require_empty)

    def _rename_df_map_keys_to_pxl_split_names(self, df_map: map):
        super()._rename_df_map_keys_to_pxl_split_names(df_map)

    def _save_binary_image(self, image_folder: str, image_name: str, binary_image):
        super()._save_binary_image(image_folder, image_name, binary_image)

    def _save_image_from_path(self, image_folder: str, image_name: str, image_path):
        super()._save_image_from_path(image_folder, image_name, image_path)

