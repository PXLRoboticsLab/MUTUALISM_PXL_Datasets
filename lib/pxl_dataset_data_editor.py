import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pandas as pd
import numpy as np
import os
import shutil

class PXL_Dataset_Data_Editor:
    def __init__(self, master, df, save_directory, start_index:int=0):
        """
        Initialize the main components of the dataset editor GUI.
        """
        self.master = master
        self.master.title("Object Detection Dataset Editor")
        self.df = df
        self.save_directory = save_directory
        self.current_image_index = start_index

        if not (os.path.exists(self.save_directory)):
            os.mkdir(self.save_directory)

        # List to store rectangle coordinates
        self.rectangles = []
        self.active_rectangle_index = tk.IntVar(value=0)  # Variable to keep track of the active rectangle

        
        # Setup image
        image_path = df.iloc[self.current_image_index]["image"]
        objects = self.df.iloc[self.current_image_index]["objects"]
        self.load_image(image_path)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_image))
        self.image_label = tk.Label(master, image=self.photo)
        self.image_label.grid(row=1, column=1)
        self.setup_sliders()
        self.load_objects_as_rectangle(objects)
        self.select_rectangle(0)
        self.update_display()


        # Divider
        self.divider = tk.Frame(master, height=2, bd=1, relief="sunken")
        self.divider.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)

        # Button Frame
        self.button_frame = tk.Frame(master)
        self.button_frame.grid(row=4, column=0, columnspan=3, sticky="ew")
        
        # Buttons to add and remove rectangles
        self.add_button = tk.Button(self.button_frame, text="+", command=self.add_rectangle)
        self.add_button.pack(side=tk.LEFT, padx=10)
        self.remove_button = tk.Button(self.button_frame, text="Remove Box", command=self.remove_rectangle)
        self.remove_button.pack(side=tk.LEFT, padx=10)
        self.next_button = tk.Button(self.button_frame, text="Next Image", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=10)

        # Container for radio buttons
        self.radio_frame = tk.Frame(master)
        self.radio_frame.grid(row=5, column=0, columnspan=3, sticky="ew")
        self.radio_buttons = []

        # Initialize radio buttons and select the first rectangle
        self.setup_radio_buttons()



    def load_image(self, path):
        """
        Load and scale an image from a file path.
        """
        self.cv_image = cv2.imread(path)
        if self.cv_image is None:
            raise FileNotFoundError("Image file not found.")
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        height = self.master.winfo_screenheight() - (self.master.winfo_screenheight()//5)
        scale_ratio = height / self.cv_image.shape[0]
        new_width = int(self.cv_image.shape[1] * scale_ratio)
        self.cv_image = cv2.resize(self.cv_image, (new_width, height))

    def next_image(self):
        """
        Load the next image and save the current annotations.
        """
        if self.current_image_index < len(self.df) - 1:
            current_image_path = self.df.iloc[self.current_image_index]["image"]
            save_file_path = "{}{}.txt".format(self.save_directory, current_image_path.split('/')[-1].split(".")[0])
            self.write_yolo_bounding_boxes_file(save_file_path)
            self.current_image_index += 1
            image_path = self.df.iloc[self.current_image_index]["image"]
            objects = self.df.iloc[self.current_image_index]["objects"]
            self.load_image(image_path)
            self.setup_sliders()
            self.load_objects_as_rectangle(objects)
            self.select_rectangle(0)
            self.setup_radio_buttons()
            self.update_display()

    def setup_sliders(self):
        """
        Set up sliders for adjusting rectangle boundaries.
        """
        self.slider_top = tk.Scale(self.master, from_=0, to=self.cv_image.shape[0], orient="vertical", width=50, sliderlength=50, showvalue=False)
        self.slider_top.grid(row=1, column=0, sticky="ns")
        self.slider_top.bind("<Motion>", self.update_active_rectangle)

        self.slider_left = tk.Scale(self.master, from_=0, to=self.cv_image.shape[1], orient="horizontal", width=50, sliderlength=50, showvalue=False)
        self.slider_left.grid(row=0, column=1, sticky="ew")
        self.slider_left.bind("<Motion>", self.update_active_rectangle)

        self.slider_bottom = tk.Scale(self.master, from_=0, to=self.cv_image.shape[0], orient="vertical", width=50, sliderlength=50, showvalue=False)
        self.slider_bottom.grid(row=1, column=2, sticky="ns")
        self.slider_bottom.bind("<Motion>", self.update_active_rectangle)

        self.slider_right = tk.Scale(self.master, from_=0, to=self.cv_image.shape[1], orient="horizontal", width=50, sliderlength=50, showvalue=False)
        self.slider_right.grid(row=2, column=1, sticky="ew")
        self.slider_right.bind("<Motion>", self.update_active_rectangle)

    def setup_radio_buttons(self):
        """
        Setup radio buttons for selecting active rectangles.
        """
        # Clear existing radio buttons
        for rb in self.radio_buttons:
            rb.destroy()
        self.radio_buttons.clear()
        # Create new radio buttons
        for i, rect in enumerate(self.rectangles):
            rb = tk.Radiobutton(self.radio_frame, text=str(i + 1), variable=self.active_rectangle_index, value=i,
                                command=lambda idx=i: self.select_rectangle(idx))
            rb.pack(side=tk.LEFT, padx=10)
            self.radio_buttons.append(rb)

    def add_rectangle(self):
        """
        Add a new rectangle and update the interface.
        """
        new_rect = {"top": 50, "left": 50, "bottom": 100, "right": 100}
        self.rectangles.append(new_rect)
        self.setup_radio_buttons()
        self.select_rectangle(len(self.rectangles) - 1)

    def remove_rectangle(self):
        """
        Remove the selected rectangle if there are multiple rectangles.
        """
        if len(self.rectangles) > 1:
            index = self.active_rectangle_index.get()
            self.rectangles.pop(index)
            self.setup_radio_buttons()
            new_index = max(0, index - 1)
            self.select_rectangle(new_index)
        else:
            print("At least one rectangle must remain.")

    def select_rectangle(self, index):
        """
        Select a rectangle, update sliders and the display.
        """
        if self.rectangles:
            self.active_rectangle_index.set(index)
            rect = self.rectangles[index]
            self.slider_top.set(rect['top'])
            self.slider_left.set(rect['left'])
            self.slider_bottom.set(rect['bottom'])
            self.slider_right.set(rect['right'])
            self.update_display()

    def update_active_rectangle(self, event):
        """
        Update the properties of the active rectangle based on slider positions.
        """
        rect = self.rectangles[self.active_rectangle_index.get()]
        rect['top'] = self.slider_top.get()
        rect['left'] = self.slider_left.get()
        rect['bottom'] = self.slider_bottom.get()
        rect['right'] = self.slider_right.get()
        self.update_display()

    def update_display(self):
        """
        Update the display with rectangles drawn on the image.
        """
        image_copy = self.cv_image.copy()
        for i, rect in enumerate(self.rectangles):
            color = (128, 128, 128) if i != self.active_rectangle_index.get() else (255, 0, 0)
            cv2.rectangle(image_copy, (rect['left'], rect['top']), (rect['right'], rect['bottom']), color, 2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_copy))
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo  # prevent garbage-collection

    def load_objects_as_rectangle(self, objects):
        """
        Load objects from a data structure into rectangles for editing.
        """
        rectangles = []
        image_height, image_width = self.cv_image.shape[:2]
        for obj in objects:
            values = []
            for i in range(1, len(obj["centerNSize"]), 2):
                values.append(float(obj["centerNSize"][i-1]) * image_width)
                values.append(float(obj["centerNSize"][i]) * image_height)
            t = int(values[1]-(values[3]/2))
            r = int(values[0]+(values[2]/2))
            l = int(values[0]-(values[2]/2))
            b = int(values[1]+(values[3]/2))
            rectangle = {"top": t, "left": l, "bottom": b, "right": r}
            rectangles.append(rectangle)
        self.rectangles = rectangles

    def write_yolo_bounding_boxes_file(self, filename:str):
        """
        Write the bounding box data in YOLO format to a file.
        """
        with open(filename, 'w') as file:
            for rectangle in self.rectangles:
                image_height, image_width = self.cv_image.shape[:2]
                center_x = float(int((rectangle['right']-rectangle['left'])/2.0) + rectangle['left'])/image_width
                center_y = float(int((rectangle['bottom']-rectangle['top'])/2.0) + rectangle['top'])/image_height
                size_x = float(rectangle['right'] - rectangle['left'])/image_width
                size_y = float(rectangle['bottom'] - rectangle['top'])/image_height
                line = "0 {} {} {} {}\n".format(center_x, center_y, size_x, size_y)
                file.write(line)


