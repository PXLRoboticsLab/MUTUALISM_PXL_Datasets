from .pxl_datasets import PXL_datasets
from .pxl_dataset_types import PXL_dataset_types
from .pxl_dataset_loader import PXL_dataset_loader
from .pxl_dataset_split import PXL_dataset_split
from .pxl_dataset_object_detection import PXL_object_detection_dataset

from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class PXL_segmentation_dataset(PXL_datasets):

    def __init__(self):
        """
        Initializes the segmentation dataset and specifies its type using the PXL_dataset_types enum.
        """
        super().__init__()
        self.dataset_type = PXL_dataset_types.Segmentation
    
    def load_from_url(self, loader: PXL_dataset_loader, url: str, save_directory: str):
        """
        Loads dataset from a URL. This method utilizes an instance of the loader class.
        """
        return super().load_from_url(loader, url, save_directory)
    
    def load_from_save_dir(self, save_directory: str):
        """
        Loads dataset from a specified directory.
        """
        self.save_directory = save_directory
        directory = Path(save_directory)
        df_map = {}
        for item in directory.iterdir():
            for split in PXL_dataset_split:
                if item.is_dir and item.name.startswith(split.value):
                    image_directory = Path("{}{}/image/".format(save_directory, split.name.lower()))
                    image_paths = []
                    segmented_image_paths = []
                    for image in image_directory.iterdir():
                        if image.is_file:
                            image_paths.append("{}{}/image/{}".format(save_directory, split.name.lower(), image.name))
                            segmented_image_paths.append("{}{}/segmented/{}".format(save_directory, split.name.lower(), image.name))
                    df = pd.DataFrame({'image': image_paths, 'segmentation_image': segmented_image_paths})
                    df_map[split.name.lower()] = df
        self.df_map = df_map
        return self.df_map
    
    def load_from_df_map(self, df_map:map, save_directory:str, loader:PXL_dataset_loader=PXL_dataset_loader()):
        """
        Loads dataset from a dataframe map. This method allows the reuse of existing dataframe maps to
        reconstruct the dataset in a new instance.
        """
        return super().load_from_df_map(df_map, save_directory, loader)
    
    def print_dataset_information(self):
        """
        Print a summary of the dataset information.
        """
        super().print_dataset_information()
    
    def display_random_image(self, split:PXL_dataset_split):
        """
        Displays a random image and its corresponding segmentation from a given dataset split.
        """
        df = self.df_map[split.name.lower()]
        row = df.sample(n=1).iloc[0]
        image = cv2.imread(row['image'])
        segmented_image = cv2.imread(row['segmentation_image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.axis('off')

    def convert_to_object_detection_dataset(self, save_directory:str, loader:PXL_dataset_loader=PXL_dataset_loader()):
        """
        Converts a segmentation dataset into an object detection dataset by computing bounding boxes
        from the segmentation masks.
        """
        print("Converting images into bounding boxes...")
        for key in self.df_map.keys():
            all_boxes = []
            for index, row in self.df_map.get(key).iterrows():
                segmented_image = row['segmentation_image']
                boxes = self._get_boundingboxes_from_segmentation(segmented_image)
                all_boxes.append(boxes)
            self.df_map[key]['objects'] = pd.Series(all_boxes, index=self.df_map[key].index)
            self.df_map[key].drop('segmentation_image', axis=1)

        new_dataset = PXL_object_detection_dataset()
        new_dataset.load_from_df_map(self.df_map, save_directory, loader)
        return new_dataset


    def _get_boundingboxes_from_segmentation(self, path:str):
        """
        Private method to extract bounding boxes from a segmented image.
        """
        if not cv2.cuda.getCudaEnabledDeviceCount():
            image = cv2.imread(path)
            image_height, image_width = image.shape[:2]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        else: # not yet integrated in the normal opencv installation version but can already be compiled
            gpu = cv2.cuda_GpuMat()
            image = cv2.imread(path)
            gpu.upload(image)
            gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
            _, gpu = cv2.cuda.threshold(gpu, 1, 255, cv2.THRESH_BINARY)
            binary_image = gpu.download()
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x = (float(x) + (float(w)/2)) / image_width
            y = float(y + (float(h)/2)) / image_height
            w = float(w) / image_width
            h = float(h) / image_height
            boxes.append({"centerNSize": [x, y, w, h]})
        return boxes
