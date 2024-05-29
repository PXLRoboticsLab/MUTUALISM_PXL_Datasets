import concurrent.futures
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
from pathlib import Path
import tkinter as tk
import json

from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_split import PXL_dataset_split
from .pxl_dataset_types import PXL_dataset_types
from .pxl_datasets import PXL_datasets
from .pxl_dataset_data_editor import PXL_Dataset_Data_Editor


class PXL_object_detection_dataset(PXL_datasets):
    """
    Extendedclassof PXL_datasets, handling specific operations for object detection datasets.
    """
    def __init__(self):
        """
        Initiolising class and setting the dataset_type to Object_Detection.
        """
        super().__init__()
        self.dataset_type = PXL_dataset_types.Object_Detection
    
    def load_from_url(self, loader: PXL_dataset_loader, url: str, save_directory: str):
        """
        Load dataset from a URL using the specified loader.
        """
        return super().load_from_url(loader, url, save_directory)
    
    def load_from_save_dir(self, save_directory: str, loader: PXL_dataset_loader=PXL_dataset_loader()):
        """
        Load dataset from a local save directory.
        """
        self.save_directory = save_directory
        self.loader = loader
        directory = Path(save_directory)
        df_map = {}
        for item in directory.iterdir():
            for split in PXL_dataset_split:
                if item.is_dir and item.name.startswith(split.value):
                    directory = Path("{}{}/".format(save_directory, split.name.lower()))
                    image_paths = []
                    for image in directory.iterdir():
                        if image.is_file and image.suffix != ".txt":
                            image_paths.append("{}{}/{}".format(save_directory, split.name.lower(), image.name))
                    df = pd.DataFrame({'image': image_paths})
                    df['objects'] = None
                    df_map[split.name.lower()] = df
        for key in df_map.keys():
            for index, row in df_map.get(key).iterrows():
                filename = "{}.txt".format(row['image'].split('.')[0])
                objects = self._load_yolo_txt_file(filename)
                df_map[key].at[index, 'objects'] = objects
        self.df_map = df_map
        return self.df_map
    
    def load_from_df_map(self, df_map:map, save_directory:str, loader:PXL_dataset_loader=PXL_dataset_loader()):
        """
        Load dataset from an existing DataFrame map.
        """
        return super().load_from_df_map(df_map, save_directory, loader)
    
    def display_random_image(self, split:PXL_dataset_split):
        """
        Display a random image from the specified dataset split.
        """
        df = self.df_map[split.name.lower()]
        row = df.sample(n=1).iloc[0]
        self._display_image_with_objects(row['image'], row['objects'])

    def replace_object_files(self, source_directory:str, keep_other_directory:str=None):
        """
        Replace object files from source directory, optionally keeping backups.
        """
        if(os.path.exists(keep_other_directory)):
            shutil.rmtree(keep_other_directory)
        os.mkdir(keep_other_directory)
        source_directory = Path(source_directory)
        for source_item in source_directory.iterdir():
            if source_item.is_file() and source_item.name.endswith(".txt"):
                for split in PXL_dataset_split:
                    path_proposal = "{}{}/{}".format(self.save_directory, split.name.lower(), source_item.name)
                    if(os.path.exists(path_proposal)):
                        if(keep_other_directory != None):
                            keep_other_filename = "{}{}".format(keep_other_directory, source_item.name)
                            shutil.copyfile(source_item, keep_other_filename)

                        os.remove(path_proposal)
                        shutil.copyfile(source_item, path_proposal)
        for key in self.df_map.keys():
            for index, row in self.df_map.get(key).iterrows():
                filename = "{}.txt".format(row['image'].split('.')[0])
                objects = self._load_yolo_txt_file(filename)
                self.df_map[key].at[index, 'objects'] = objects

    def manual_improve_data(self, df, save_directory:str, continue_index:int=0):
        """
        Manually improve data using a Tkinter interface.
        """
        root = tk.Tk()
        app = PXL_Dataset_Data_Editor(root, df, save_directory, continue_index)
        root.mainloop()

    def export_dataset_in_COCO_format(self, save_directory:str):
        """
        Export the dataset in the famous COCO format for training
        """
        if(os.path.exists(save_directory)):
            shutil.rmtree(save_directory)
        os.mkdir(save_directory)
        for split in PXL_dataset_split:
            if (split.name.lower() in self.df_map.keys()):
                os.mkdir("{}{}".format(save_directory, split.name.lower()))
                images = []
                annotations = []
                categories = set()
                annotation_id = 0
                for index, row in self.df_map[split.name.lower()].iterrows():
                    filename = row['image'].split("/")[-1].replace(".png", ".jpg")
                    image = cv2.imread(row['image'])
                    image_height, image_width = image.shape[:2]
                    images.append({
                        "id": index,
                        "width": image_width,
                        "height": image_height,
                        "file_name": filename
                    })
                    save_filename = "{}{}/{}".format(save_directory, split.name.lower(), filename)
                    shutil.copyfile(row['image'], save_filename)
                    for obj in row['objects']:
                        class_id = obj.get('name', 0)
                        values = []
                        for i in range(1, len(obj["centerNSize"]), 2):
                            values.append(float(obj["centerNSize"][i-1]) * image_width)
                            values.append(float(obj["centerNSize"][i]) * image_height)
                        x_min = int(values[0]-(values[2]/2))
                        y_min = int(values[1]-(values[3]/2))
                        w = values[2]
                        h = values[3]
                        annotations.append({
                                "id": annotation_id,
                                "image_id": index,
                                "category_id": int(class_id),
                                "bbox": [x_min, y_min, w, h],
                                "area": w * h,
                                "iscrowd": 0,
                                "segmentation": []
                        })
                        annotation_id += 1

                # Convert categories set to list of dicts
                categories = [{"id": 0, "name": 0}]

                # Prepare the final dictionary to be written as JSON
                dataset_file = {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories
                }

                dataset_file_path = "{}{}/dataset.json".format(save_directory, split.name.lower())
                # Write the COCO data to a JSON file
                with open(dataset_file_path, 'w') as json_file:
                    json.dump(dataset_file, json_file, indent=4)


    def print_dataset_information(self):
        """
        Print a summary of the dataset information.
        """
        super().print_dataset_information()
        print("Number of objects per split:")
        for split in PXL_dataset_split:
            print("\t{}:".format(split.name))
            total_objects = 0
            for index, row in self.df_map[split.name.lower()].iterrows():
                total_objects += len(row["objects"])
            print("\t\t{}".format(total_objects))
        

    def _load_yolo_txt_file(self, filepath):
        """
        Private function to load annotations from a YOLO format text file.
        """
        with open(filepath, 'r') as file:
            boxes = []
            lines = file.readlines()
            for line in lines:
                split_line = line.strip().split(" ")
                box_class = split_line.pop(0)
                poly = []
                if len(split_line) > 4:
                    for i in range(1, len(split_line), 2):
                        poly.append([split_line[i-1], split_line[i]])
                    boxes.append({"name": box_class, "poly": poly})
                else:
                    boxes.append({"name": box_class, "centerNSize": split_line})
        return boxes
    
    def _display_image_with_objects(self, path, objects):
        """
        Private function to display an image with annotated objects overlayed.
        """
        image = cv2.imread(path)
        image_height, image_width = image.shape[:2]
        plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for obj in objects:
            values = []
            for i in range(1, len(obj["centerNSize"]), 2):
                values.append(float(obj["centerNSize"][i-1]) * image_width)
                values.append(float(obj["centerNSize"][i]) * image_height)
            tl = [int(values[0]-(values[2]/2)), int(values[1]-(values[3]/2))]
            tr = [int(values[0]+(values[2]/2)), int(values[1]-(values[3]/2))]
            bl = [int(values[0]-(values[2]/2)), int(values[1]+(values[3]/2))]
            br = [int(values[0]+(values[2]/2)), int(values[1]+(values[3]/2))]
            rectangle = [tl, tr, br, bl]
            points = np.array([rectangle], np.int32)
            plt_image = cv2.polylines(plt_image, [points], isClosed=True, color=(255,255,0), thickness=int(image_width/200.0))
        plt.figure(figsize=(10, 10))
        plt.imshow(plt_image)
        plt.axis('off')
        plt.title(path)
        plt.show()
