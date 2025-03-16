from PIL import Image
import numpy as np
import torch


label_encoder = {
                    'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Van': 3,
                    'Person_sitting': 4, 'Truck': 5, 'Tram': 6, 'Misc': 7,
                    'DontCare': -1
                }


def load_image(path: str) -> Image:
    """
        Loads an image as grayscale (using Pillow).
        Note: do not normalize the image to [0,1]
        Args:
            paths: Tuple containing the path to image and annotation.
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
    """
    image = Image.open(path)#.convert(mode='L')

    return image


def load_training_labels(labelPath, num_classes):
    label_id = []
    boxes = []
    with open(labelPath, "r") as f:
        for line in f:
            lineContents = line.split("\n")[0].split(" ")
            classLabel = lineContents[0]
            if num_classes == 8:
                if classLabel != "DontCare":
                    label_id.append(label_encoder[classLabel])
                    boxes.append([lineContents[4], lineContents[5], lineContents[6], lineContents[7]])
            elif num_classes == 3:
                if classLabel in ["Car", "Pedestrian", "Cyclist"]:
                    label_id.append(label_encoder[classLabel])
                    boxes.append([lineContents[4], lineContents[5], lineContents[6], lineContents[7]])
            else:
                raise ValueError("Number of classes must either be 8 or 3")
    f.close()
    label_tensor = torch.tensor(label_id)
    box_tensor = torch.tensor(np.array(boxes).astype(float).astype(int))
    
    return box_tensor, label_tensor