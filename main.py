import cv2
import numpy as np
import argparse
from gen import CaptchaSequence
from model import Model

def train(dataset):
    Gen_Train = CaptchaSequence(batch_size=8, steps=1, dataset=dataset)
    Gen_Valid = CaptchaSequence(batch_size=8, steps=1, dataset=dataset)
    print('Generator Building Complete')
    model = Model(model=None)
    model.train(Gen_Train, Gen_Valid)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=False, type=str, default='python_captcha')
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-a", "--attack", action="store_true")

    args = parser.parse_args()

    if(args.train):
        train(args.dataset)
    
    
