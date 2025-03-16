import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from src.plots.bounding_box import image_with_bounding_box
import matplotlib.animation as animation
from src.models.retinanet.resnet import *

model = resnet152(num_classes=9, pretrained=True)

h, w = 384, 1248
image_tensor = torch.zeros(3, 384, 1248)
image = to_pil_image(image_tensor)
image_tensor = torch.unsqueeze(image_tensor, 0)
anchors = model(image_tensor)[3].squeeze(0)
width = anchors[:, 2] - anchors[:, 0]
height = anchors[:, 3] - anchors[:, 1]
increment = 1
gif_images = []

fig, ax = plt.subplots()
fig.set_size_inches(31.2, 9.6)

def plotAnchors(increment):
    i = 0
    while i < len(anchors):
        selectAnchors = anchors[i:min(i + increment, len(anchors))]
        bbImage = image_with_bounding_box(image, selectAnchors, None, "white")
        im = ax.imshow(bbImage)
        ax.axis("off")
        gif_images.append([im])
        i += increment
    ani = animation.ArtistAnimation(fig, gif_images, blit=True)
    ani.save("anchors.mp4")