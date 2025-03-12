from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def image_with_bounding_box(image, boxes):
    overlaid_image =  draw_bounding_boxes(image=image, boxes=boxes, width=5)

    return to_pil_image(overlaid_image)