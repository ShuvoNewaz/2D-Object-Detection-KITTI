from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch


def image_with_bounding_box(image, boxes, labels, colors):
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