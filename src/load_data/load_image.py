import os
import cv2

def load_image(path: str) -> tuple[list, list]:
    image_list = []
    label_list = []

    for entry in os.listdir(path):    
        full_path = os.path.join(path, entry)

        image_list.append(cv2.imread(full_path))
        label_list.append(int(entry.split('_', 1)[0]))

    return image_list, label_list
