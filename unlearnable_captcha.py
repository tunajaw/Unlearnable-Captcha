import cv2
import numpy as np
import string
from captcha_seqence import CaptchaSequence
from modelA import modelA
from modelB import modelB
from modelC import modelC
from attack_model import attack_Model
from tqdm import tqdm
import sys
#import skimage
#from skimage import data
#from skimage import transform


class unlearnable_captcha():
    def __init__(self, height=64, width=128, n_len=4, n_class=36, custom_string=None) -> None:
        # constants
        self.IMPLEMENTED_MODELS = ('modelA', 'modelB', 'modelC')
        self.PRETRAINED_MODEL_PATH = {
            'modelA': './pretrained/cnn_best.h5',
            'modelB': './pretrained/cnn_bestB.h5'
        }
        # variables
        self.height = height
        self.width = width
        self.n_len = n_len
        self.proxy_model = None 
        self.cur_attack_model = None
        self.n_class = n_class
        self.custom_string = custom_string
        # to-do : Not sure initialize dataset
        if(self.custom_string is not None):
            self.dataset = 'custom'
        else:
            self.dataset = 'python_captcha'
        #self.dataset =skimage.transform.resize(self.dataset,(64,128))


    def train(self, batch_size=128, dataset=None, model='modelA') -> None:
        if(model not in self.IMPLEMENTED_MODELS):
            raise ValueError(f'{model} is not implemented.')
        self.dataset = dataset
        Gen_Train = CaptchaSequence(batch_size=batch_size, steps=1000, dataset=dataset)
        Gen_Valid = CaptchaSequence(batch_size=batch_size, steps=100, dataset=dataset)
        self.proxy_model = getattr(sys.modules[__name__], str(model))(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        # self.proxy_model = modelB(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.proxy_model.train(Gen_Train, Gen_Valid)

    def load_proxy_model(self, model='modelA', test=False) -> None:
        # load pretrained model
        print(f'load pretrained proxy model: {model}')
        if(model not in self.IMPLEMENTED_MODELS):
                raise ValueError(f'{model} is not implemented.')
        self.proxy_model = getattr(sys.modules[__name__], str(model))(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.proxy_model.load_model(self.PRETRAINED_MODEL_PATH[model])
        # test pretrained model
        if(test):
            Gen = CaptchaSequence(batch_size=1, steps=1, dataset=self.dataset, custom_string=self.custom_string)
            test_img, test_y = Gen[0]
            # test if model is 
            test_pred = self._proxy_model_predict(test_img)
            # decode one-hot label back to string
            test_y = self._decode(test_y)
            print(f'ground truth: {test_y}')
            print(f'predicted: {test_pred}')

    def load_attacked_model(self, model):
        print(f'load pretrained attacked model: {model}')
        if(model not in self.IMPLEMENTED_MODELS):
                raise ValueError(f'{model} is not implemented.')
        self.cur_attack_model = getattr(sys.modules[__name__], str(model))(height=self.height, width=self.width, n_len=self.n_len, _model=None)
        self.cur_attack_model.load_model(self.PRETRAINED_MODEL_PATH[model])

    def _proxy_model_predict(self, X) -> list:
        return self.proxy_model.predict(X)

    def _decode(self, y) -> list:
        return self.proxy_model.decode(y)

    def gen_attack_img(self, gen_imgs=1, method='iFGSM', iter_atk=True):
        '''
        generate adversarial attack images.

        '''
        attack_model = attack_Model(self.n_class)
        test_time = gen_imgs
        Gen = CaptchaSequence(batch_size=test_time, steps=1, dataset=self.dataset, custom_string=self.custom_string)
        test_img, one_hot_y = Gen[0]
        
        # decode one-hot label back to string
        test_y = self._decode(one_hot_y)
        # cv2.imwrite('ori3.jpg', np.array(test_img[0]*255).astype(np.uint8))
        # cv2.imwrite('ori4.jpg', np.array(test_img[1]*255).astype(np.uint8))
        a_img = attack_model.attack(method, test_img, test_y, one_hot_y, self.proxy_model, iterative=iter_atk)
        # test if model is 
        test_pred = self._proxy_model_predict(test_img)
        # print(test_pred)
        return test_img, a_img, test_y
    
    def attack(self, attacked_model=['model_B'], imgs=100, method='iFGSM', iter_atk=True, test_img_show=True):
        for amodel in attacked_model:
            self.load_attacked_model(str(amodel))
            s, f = 0, 0
            test_time = imgs
            for _ in tqdm(range(100)):
                ori_img, a_img, test_y = self.gen_attack_img(gen_imgs=1, method=method, iter_atk=iter_atk)
                if(test_img_show):
                    cv2.imwrite('ori.jpg', np.array(ori_img[0]*255).astype(np.uint8))
                    cv2.imwrite('att.jpg', np.array(a_img[0]*255).astype(np.uint8))
                test_pred = self.cur_attack_model.predict(ori_img)
                a_pred = self.cur_attack_model.predict(a_img)
                print(f'ground truth: {test_y}')
                print(f'predicted: {test_pred}')
                print(f'after_attack: {a_pred}')
                print('--------------')
                if((test_pred[0]==test_y[0]) and (test_pred[0]!=a_pred[0])): s += 1
                elif((test_pred[0]==test_y[0]) and (test_pred[0]==a_pred[0])): f += 1

            print(f'proxy model accuracy: {(s+f)/test_time*100}%')
            print(f'attack success: {s}/{s+f}, {100*s/(s+f):.2f}%')
            print(f'attack failed: {f}/{s+f}, {100*f/(s+f):.2f}%')
