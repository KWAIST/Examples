# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:15:43 2024

@author: jaege
"""

import tkinter as tk
from PIL import Image, ImageTk
import json
from tkinter import filedialog

label = ""
box = []

class ImageLabelViewer:
    def __init__(self, root, image_folder, label_file):
        self.root = root
        self.image_folder = image_folder
        self.label_file = label_file

        self.images = []
        self.labels = []
        self.boxes = []  # Initialize as an empty list

        self.photo_images = []  # List to store references to PhotoImage objects
        self.current_index = 0
        label = ""
        box = []

        self.load_data()

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.update_image()

        root.bind('<Right>', self.next_image)
        root.bind('<Left>', self.prev_image)

    def update_image(self):
        image = ImageTk.PhotoImage(self.images[self.current_index])
        self.photo_images.append(image)  # Store reference in the list
        self.canvas.config(width=image.width(), height=image.height())
        self.canvas.create_image(0, 0, anchor='nw', image=image)
        self.canvas.image = image

        label_text = f"Class: {self.labels[self.current_index]}"
        self.canvas.create_text(10, 10, anchor='nw', text=label_text, fill='white', font=('Arial', 12))

        box = self.boxes[self.current_index]
        x, y, width, height = box
        self.canvas.create_rectangle(x, y, x + width, y + height, outline='red', width=2)
        print("Class : ", label_text, " Location : ", box) 
        
    def load_data(self):
        with open(self.label_file, 'r') as f:
            data = json.load(f)

        for item in data['categories']:
            label = item['name']
        
        for item in data['annotations']:
            box = item['bbox']
            id = item['image_id']

            for item1 in data['images']:
                image_path = f"{self.image_folder}/{item1['file_name']}"
                if id == item1['id']:
                    self.images.append(Image.open(image_path))
                    self.labels.append(label)
                    self.boxes.append(box)
        
    def next_image(self, event):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.update_image()

    def prev_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Labeled Image Viewer")

    image_folder_path = '/Users/Jaege/TestPGM/Images'
    label_json_path = '/Users/Jaege/TestPGM/Images/coco.json'
 
    viewer = ImageLabelViewer(root, image_folder_path, label_json_path)

    root.mainloop()