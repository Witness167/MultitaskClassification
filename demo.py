from __future__ import print_function, division

import torch
from torch.nn import functional as F
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import logging

from mtcnn.detector import detect_faces
from align_faces import warp_and_crop_face, get_reference_facial_points
from model import MultiTaskNet


def image_loader(image_path, out_size, device, transform=None):
    global dst_img

    raw = cv.imread(image_path)
    img = Image.open(image_path).convert('RGB')
    _, facial5points = detect_faces(img)

    if len(facial5points) != 1:
        raise Exception("No face or multi faces...")

    facial5points = np.reshape(facial5points[0], (2, 5))

    # Default set
    crop_size = 224
    inner_padding_factor = 0.1
    outer_padding = 0
    output_size = 224

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        (output_size, output_size), inner_padding_factor, (outer_padding, outer_padding), True)

    dst_img = warp_and_crop_face(
        raw, facial5points, reference_pts=reference_5pts, crop_size=(
            crop_size, crop_size)
    )
    dst_img = cv.resize(dst_img, (out_size, out_size))[:, :, ::-1]
    dst_img_ = Image.fromarray(dst_img)
    im = transform(dst_img_).float()

    return im.unsqueeze(0).to(device)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Loading model
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    model = MultiTaskNet(model_name='facenet', num_embeddings=256)
    checkpoint = torch.load("tensorboard/epoch06_0.274_0.430.pth", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    img = image_loader(image_path='data/images/AF1.png', out_size=160, device=device, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])]))
    face_attribute = ['Face', 'Mouth', 'Eyebrow', 'Eye', 'Nose', 'Jaw']
    
    with torch.no_grad():
        outputs = model(img)

        for i in range(len(face_attribute)):
            logging.info('{}Grade: {}'.format(face_attribute[i],
                                              torch.max(F.softmax(outputs[i], dim=1), dim=-1)[1].item()))

        plt.imshow(dst_img)
        plt.show()
