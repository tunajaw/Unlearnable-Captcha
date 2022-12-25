import numpy as np
import string
from captcha.image import ImageCaptcha
import random
import cv2

class CaptchaGen():
    def __init__(self, dataset=None, width=128, height=64, n_len=4) -> None:
        self.dataset = dataset
        self.n_len = n_len
        if(self.dataset == 'python_captcha'):
            self.generator = ImageCaptcha(width, height)
        else:
            raise ValueError(f'{self.dataset} is not a valid dataset.')

    # return (image, label)
    def generate_image(self, idx) -> tuple([np.array, string]):
        if(self.dataset == 'python_captcha'):
            characters = string.digits + string.ascii_uppercase
            random_str = ''.join([random.choice(characters) for j in range(self.n_len)])
            return self.generator.generate_image(random_str), random_str
        elif(self.dataset == 'w'):
            0
        else:
            raise ValueError(f'{self.dataset} is not a valid dataset.')