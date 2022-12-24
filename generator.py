# Ref: https://github.com/ypwhs/captcha_break

import numpy as np
import random
import tensorflow as tf
import keras.backend as K
from keras.utils import Sequence
from .single_gen import ImageCaptcha

class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64, dataset=None):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(dataset=dataset, width=width, height=height)
    
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            X[i], label = np.array(self.generator.generate_image()) / 255.0
            for j, ch in enumerate(label):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y