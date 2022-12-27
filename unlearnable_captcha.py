import cv2
import numpy as np
import string
from captcha_seqence import CaptchaSequence
from model import model

class unlearnable_captcha():
    def __init__(self, height=64, width=128, n_len=4) -> None:
        self.height = height
        self.width = width
        self.n_len = n_len
        self.proxy_model = None
        # to-do : Not sure initialize dataset
        self.dataset = 'python_captcha'


    def train(self, batch_size=128, dataset=None) -> None:
        self.dataset = dataset
        Gen_Train = CaptchaSequence(batch_size=batch_size, steps=1000, dataset=dataset)
        Gen_Valid = CaptchaSequence(batch_size=batch_size, steps=100, dataset=dataset)
        self.proxy_model = model(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.proxy_model.train(Gen_Train, Gen_Valid)

    def load_proxy_model(self, model_path='./pretrained/cnn_best.h5', test=True) -> None:
        # load pretrained model
        self.proxy_model = model(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.proxy_model.load_model(model_path)
        # test pretrained model
        if(test):
            Gen = CaptchaSequence(batch_size=1, steps=1, dataset=self.dataset)
            test_img, test_y = Gen[0]
            # test if model is 
            test_pred = self._proxy_model_predict(test_img)
            # decode one-hot label back to string
            test_y = self._decode(test_y)
            print(test_y)
            print(test_pred)

    def _proxy_model_predict(self, X) -> list:
        return self.proxy_model.predict(X)

    def _decode(self, y) -> list:
        return self.proxy_model.decode(y)

    def attack(self):
        return 0
