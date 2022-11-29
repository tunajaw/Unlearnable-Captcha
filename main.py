import cv2
import numpy as np
import argparse
from gen import CaptchaSequence
from model import model

def train(dataset, batch_size=128):
    Gen_Train = CaptchaSequence(batch_size=batch_size, steps=1000, dataset=dataset)
    Gen_Valid = CaptchaSequence(batch_size=batch_size, steps=100, dataset=dataset)
    print('Generator Building Complete')
    proxy_model = model(model=None)
    print('strart training')
    proxy_model.train(Gen_Train, Gen_Valid)
    print('complete')

def attack():
    return 0
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=False, type=str, default='python_captcha')
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-a", "--attack", action="store_true")

    args = parser.parse_args()

    if(args.train):
        train(args.dataset)
    
    
