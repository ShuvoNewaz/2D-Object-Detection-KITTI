from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch


def image_with_bounding_box(image, boxes, class_labels):
    label_map = {0: 'Car', 1: 'Ped', 2: 'Cyc', 3: 'Van',
                4: 'PS', 5: 'Truck', 6: 'Tram', 7: 'Misc', -1: ""}
    color_map = {0: "red", 1: "green", 2: "black", 3: "cyan",
                  4: "blue", 5: "yellow", 6: "orange", 7: "purple", -1: "white"}
    labels, colors = [], []
    for class_label in class_labels:
        labels.append(label_map[class_label.item()])
        colors.append(color_map[class_label.item()])

    boxes = boxes.long()
    if not torch.is_tensor(image):
        image = pil_to_tensor(image)
    if len(boxes) == 0:
        return to_pil_image(image)
    overlaid_image =  draw_bounding_boxes(image=image, boxes=boxes,
                                          labels=labels, colors=colors,
                                          width=5, font_size=30,
                                          font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

    return to_pil_image(overlaid_image)