from .pxl_dataset_sources import PXL_dataset_sources
from .pxl_dataset_split import PXL_dataset_split
from .pxl_dataset_types import PXL_dataset_types
from .pxl_value_exception import PXL_value_exception

import os
import time
import shutil
import io
from PIL import Image
from pathlib import Path
import concurrent.futures


class PXL_dataset_loader(object):
    """
    Class to handle the loading, saving, and preprocessing of datasets within the module.
    This includes tasks such as downloading datasets, saving them to a directory, and manipulating
    dataset directories and files according to specific types of datasets.
    """
    def download_dataset(self, url: str, dataset_source: PXL_dataset_sources, dataset_name: str):
        """
        Abstract method for downloading the dataset. Should be implemented at each datasource.
        """
        pass

    def save_dataset(self, dataset_type: PXL_dataset_types, df_map: map, save_directory: str):
        """
        Saves a dataset to a specified directory adjusted to the dataset type.
        """
        self._prepare_save_directory(save_directory)
        if dataset_type == PXL_dataset_types.Object_Detection:
            for key in df_map.keys():
                path = "{}{}/".format(save_directory, key)
                print("saving into: ", path)
                os.makedirs(path)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for index, row in df_map.get(key).iterrows():
                        objects_filename = "{}{}/{}.txt".format(save_directory, key, row['image'].split('/')[-1].split('.')[0])
                        target_image_filename = "{}{}/{}".format(save_directory, key, row['image'].split('/')[-1])
                        executor.submit(self._write_yolo_bounding_boxes_file, objects_filename, row['objects'])
                        executor.submit(shutil.copyfile, row['image'], target_image_filename)
            df_map = self._change_parent_directory_of_image_path(df_map, save_directory)
            return df_map

        elif dataset_type == PXL_dataset_types.Segmentation:            
            for key in df_map.keys():
                path = "{}{}/".format(save_directory, key)
                print("saving into: ", path)
                os.makedirs("{}image/".format(path))
                os.makedirs("{}segmented/".format(path))
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for index, row in df_map.get(key).iterrows():
                        image_path = row.get('image')
                        image_name = image_path.split("/")[-1].split(".")[0]
                        output_path = "{}{}.png".format(path + "image/", image_name)
                        executor.submit(shutil.copyfile, image_path, output_path)
                        image_path = row.get('segmentation_image')
                        image_name = image_path.split("/")[-1].split(".")[0]
                        output_path = "{}{}.png".format(path + "segmented/", image_name)
                        executor.submit(shutil.copyfile, image_path, output_path)
            df_map = self._change_parent_directory_of_image_path(df_map, save_directory)
            return df_map

        elif dataset_type == PXL_dataset_types.Classification:
            pass
        else:
            PXL_value_exception("Dataset type not implemented yet")

    def _prepare_save_directory(self, path: str, require_empty=True):
        """
        Private method that prepares the save directory by ensuring it exists and is empty.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        elif require_empty and len(os.listdir(path)) > 0:
            print("!!! It seems that de target directory is not empty! Do you want to continue and clear the directory? (y/N):")
            time.sleep(1) #wait for print te be printed
            answer = input()
            if answer == "y" or answer == "Y":
                shutil.rmtree(path)

    def _add_single_folders_to_path(self, path: str):
        """
        private method that automaticly adds single folders to the path to ensure the right path.
        """
        directory = Path(path)
        folder_content = sorted(directory.iterdir())
        while (len(folder_content) == 1):
            path += '/' + folder_content[0].name
            directory = Path(path)
            folder_content = sorted(directory.iterdir())
        return path

    def _check_and_rename_split_folders(self, path: str):
        """
        Private method that renames the dataset split folders to the correct names.
        """
        return_value = False
        directory = Path(path)
        for item in directory.interdir():
            for split in PXL_dataset_split:
                if item.startwith(split.value):
                    return_value = True
                    current_path = os.path.join(path, item)
                    new_directory_name = split.name
                    new_path = os.path.join(path, new_directory_name)
                    os.rename(current_path, new_path)
        return return_value

    def _rename_df_map_keys_to_pxl_split_names(self, df_map: map):
        """
        Private method that renames the df_map keys to the correct split names.
        """
        new_df_map = {}
        for key in df_map.keys():
            if any(key.startswith(split.value) for split in PXL_dataset_sources):
                for split in PXL_dataset_sources:
                    if key.startswith(split.value):
                        new_df_map[split] = df_map[key]
            else:
                new_df_map[key] = df_map[key]
        return new_df_map
    
    def _save_binary_image(self, image_folder: str, image_name: str, binary_image):
        """
        Private method that saves a binairy images as an image of another type.
        """
        new_image_path = image_folder + image_name
        stream = io.BytesIO(binary_image)
        image = Image.open(stream)
        image.save(new_image_path)

    def _save_image_from_path(self, image_folder: str, image_name: str, image_path: str):
        """
        Private method that copies images (and change the type) to another folder.
        """
        new_image_path = image_folder + image_name
        image = Image.open(image_path)
        image.save(new_image_path)

    def _resave_images_as_png(self, image_path, path):
        """
        Private method to save an image as a png.
        """
        image_name = image_path.split("/")[-1].split(".")[0]
        output_path = "{}{}.png".format(path, image_name)
        if(image_path.lower().endswith('.png')):
            shutil.copyfile(image_path, output_path)
        else:
            with Image.open(image_path) as image:
                image.save(output_path, 'PNG')
    
    def _change_parent_directory_of_image_path(self, df_map, new_parent_directory):
        """
        Private method to change the parent directory of the images paths to another one.
        """
        for key in df_map.keys():
            df_map.get(key)['image'] = df_map.get(key)['image'].apply(lambda x: "{}{}/image/{}".format(new_parent_directory, key, x.split('/')[-1]))
            if 'segmentation_image' in df_map.get(key).columns:
                df_map.get(key)['segmentation_image'] = df_map.get(key)['segmentation_image'].apply(lambda x: "{}{}/segmented/{}".format(new_parent_directory, key, x.split('/')[-1]))
        return df_map
    
    def _write_yolo_bounding_boxes_file(self, filename:str, boxes):
        """
        Private method to save objects to a yolo fileformat.
        """
        with open(filename, 'w') as file:
            for box in boxes:
                line = "0 "
                for point in box['centerNSize']:
                    line += str(point) + " "
                line += '\n'
                file.write(line)