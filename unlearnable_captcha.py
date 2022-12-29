import cv2
import numpy as np
import string
from captcha_seqence import CaptchaSequence
from modelA import modelA
from attack_model import attack_Model
from tqdm import tqdm
import skimage
#from skimage import data
#from skimage import transform


class unlearnable_captcha():
    def __init__(self, height=64, width=128, n_len=4, n_class=36) -> None:
        self.height = height
        self.width = width
        self.n_len = n_len
        self.proxy_model = None
        self.n_class = n_class
        # to-do : Not sure initialize dataset
        self.dataset = 'python_captcha'
        #self.dataset =skimage.transform.resize(self.dataset,(64,128))


    def train(self, batch_size=128, dataset=None) -> None:
        self.dataset = dataset
        Gen_Train = CaptchaSequence(batch_size=batch_size, steps=1000, dataset=dataset)
        Gen_Valid = CaptchaSequence(batch_size=batch_size, steps=100, dataset=dataset)
        self.proxy_model = modelA(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.proxy_model.train(Gen_Train, Gen_Valid)

    def load_proxy_model(self, model_path='./pretrained/cnn_best.h5', test=False) -> None:
        # load pretrained model
        self.proxy_model = modelA(height=self.height, width=self.width, n_len=self.n_len, _model=None)
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
        attack_model = attack_Model(self.n_class)

        s, f = 0, 0
        test_time = 50
        for _ in tqdm(range(test_time)):

            Gen = CaptchaSequence(batch_size=1, steps=1, dataset=self.dataset)
            test_img, one_hot_y = Gen[0]
            # test if model is 
            test_pred = self._proxy_model_predict(test_img)
            # decode one-hot label back to string
            test_y = self._decode(one_hot_y)
            print(f'ground truth: {test_y}')
            print(f'predicted: {test_pred}')
            # cv2.imwrite('ori.jpg', np.array(test_img[0]*255).astype(np.uint8))
            # cv2.imwrite('ori2.jpg', np.array(test_img[1]*255).astype(np.uint8))
            a_img = attack_model.attack('iFGSM', test_img, test_y, one_hot_y, self.proxy_model, iterative=True)
            a_pred = self._proxy_model_predict(a_img)
            
            print(f'after attack: {a_pred}')
            print('------------')
            if((test_pred[0]==test_y[0]) and (test_pred[0]!=a_pred[0])): s += 1
            elif((test_pred[0]==test_y[0]) and (test_pred[0]==a_pred[0])): f += 1
            
            # print(test_img.shape)
            # cv2.imwrite('attacked.jpg', np.array(a_img[0]*255).astype(np.uint8))
            # cv2.imwrite('attacked2.jpg', np.array(a_img[1]*255).astype(np.uint8))
        
        print(f'proxy model accuracy: {(s+f)/test_time*100}%')
        print(f'attack success: {s}/{s+f}, {100*s/(s+f):.2f}%')
        print(f'attack failed: {f}/{s+f}, {100*f/(s+f):.2f}%')

