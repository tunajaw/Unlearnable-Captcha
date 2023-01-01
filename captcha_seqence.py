# Ref: https://github.com/ypwhs/captcha_break
import string
import numpy as np
import random
import tensorflow as tf
import keras.backend as K
#from keras.utils import Sequence
from tensorflow.keras.utils import Sequence
from single_gen import CaptchaGen

class CaptchaSequence(Sequence):
    def __init__(self, batch_size, steps, n_len=4, width=128, height=64, dataset=None, custom_string=None) -> None:
        self.characters = string.digits + string.ascii_uppercase
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(self.characters)
        self.custom_string = custom_string
        self.generator = CaptchaGen(dataset=dataset, width=width, height=height, n_len=self.n_len, custom_string=custom_string)
    
    # how many batches to train in a training epoch
    def __len__(self) -> int:
        return self.steps

    # get full batch of data
    def __getitem__(self, idx) -> tuple([np.array, np.array]):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            X[i], label = self.generator.generate_image(idx)
            X[i] = np.array(X[i] / 255.0)
            for j, ch in enumerate(label):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
    

        return X, y