import numpy as np
import string
from captcha.image import ImageCaptcha
import random
import cv2

class ImageCaptcha():
    def __init__(self, dataset=None, width=128, height=64, n_len=4):
        self.dataset = dataset
        self.n_len = n_len
        if(self.dataset == 'python_captcha'):
            self.generator = ImageCaptcha(width, height)
        else:
            raise ValueError(f'{self.dataset} is not a valid dataset.')

    # return (image, label)
    def generate_data(self):
        if(self.dataset == 'python_captcha'):
            characters = string.digits + string.ascii_uppercase
            random_str = ''.join([random.choice(characters) for j in range(self.n_len)])
            return self.generator.generate_images(random_str), random_str
        else:
            raise ValueError(f'{self.dataset} is not a valid dataset.')