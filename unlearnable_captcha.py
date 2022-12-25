import cv2
import numpy as np
from captcha_seqence import CaptchaSequence
from model import model
from tensorflow.keras.models import load_model

class unlearnable_captcha():
    def __init__(self, height=64, width=128, n_len=4) -> None:
        self.height = height
        self.width = width
        self.n_len = n_len
        self.proxy_model = None


    def train(self, batch_size=128, dataset=None) -> None:
        Gen_Train = CaptchaSequence(batch_size=batch_size, steps=1000, dataset=dataset)
        Gen_Valid = CaptchaSequence(batch_size=batch_size, steps=100, dataset=dataset)
        self.proxy_model = model(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.proxy_model.train(Gen_Train, Gen_Valid)

    def load_proxy_model(self, model_path='./pretrained/cnn_best.h5') -> None:
        self.proxy_model = load_model(model_path)
        

    def _proxy_model_predict(self, X) -> np.array:
        return self.proxy_model.predict(X)

    def attack(self):
        return 0
