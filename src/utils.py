import cv2
import torch
import numpy as np

def process_img(obs):

    # img_state = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    img_state = obs.reshape((obs.shape[0], obs.shape[1], 1))

    img_tensor = img_state.transpose(2, 0, 1).astype(np.float32)
    img_tensor = torch.from_numpy(img_tensor)
    return img_tensor