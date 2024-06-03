import torch
import os
import numpy as np
from typing import List, Tuple
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,  ScaleIntensityRange,
    Activations, AsDiscrete, Activations,
    Resize, EnsureType, EnsureChannelFirst,
    LoadImage
)


def load_model(model_weights: str, img_size: Tuple, spatial_dims: int, num_classes: int, in_channels: int, device):

    model = SwinUNETR(
        img_size = img_size,
        in_channels = in_channels,
        out_channels = num_classes,
        feature_size = 48,
        spatial_dims = spatial_dims,
        )
    model.load_state_dict(torch.load(model_weights)["state_dict"])
    model.eval().to(device)
    return model

def load_image(img_path: str, img_size: Tuple):
    base_image = np.load(img_path)
    base_size = base_image.shape
    
    pre_processing = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Resize(spatial_size=img_size, mode = "nearest"),
        ScaleIntensityRange(a_min=-350, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureChannelFirst(),
        EnsureType()
        ])
    image = pre_processing(img_path)
    return image, base_size

def ohe_to_classes(ohe_image: np.array, img_size: Tuple) -> np.array:
    image = np.zeros(shape = img_size)
    num_classes = ohe_image.shape[2]
    for i in range(1, num_classes):
      image[ohe_image[:,:,i] == 1] = i
    return image

def resize_image(image: np.array, base_size: Tuple) -> np.array:
    resize_transform = Resize(spatial_size=base_size, mode = "nearest")
    image = resize_transform(np.expand_dims(image, axis=0)).squeeze()
    return image
    
def post_processing(raw_predict, base_size: Tuple, img_size: Tuple) -> np.array:
    transformations = Compose([
        Activations(sigmoid=True),
        AsDiscrete(argmax=False, threshold=0.5),
    ])
    ohe_predict = transformations(raw_predict).data.cpu().numpy().squeeze().transpose((1, 2, 0))
    predict = ohe_to_classes(ohe_predict, img_size)
    base_predict = resize_image(predict, base_size)
    return base_predict

def save_result(predict: np.array, save_result_path: str):
    name_file = "result.npy"
    np.save(os.path.join(save_result_path, name_file), predict)

def run_inference(model_weights_path: str, test_data_path: str, save_result_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (256, 256)
    spatial_dims = len(img_size)
    num_classes = 8
    in_channels = 1
    
    model = load_model(model_weights_path, img_size, spatial_dims, num_classes, in_channels, device)
    image, base_size = load_image(test_data_path, img_size)
    with torch.no_grad():
        raw_predict = model(image.to(device))
    predict = post_processing(raw_predict, base_size, img_size)
    save_result(predict, save_result_path)





    
