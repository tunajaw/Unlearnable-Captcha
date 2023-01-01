
from keras.models import *
from keras.layers import *
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from IPython import display
from tqdm import tqdm
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
import numpy as np
import string
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


class modelC():
    def __init__(self, height=64, width=128, n_len=4, n_class=36, _model=None) -> None:
        '''
        CAPTCHA Break Model.

        height: height of CAPTCHA
        width: width of CAPTCHA
        n_len: sequence length of CAPTCHA
        n_class: how many possible characters in CAPTCHA
        model: which classification model to use
        '''
        self.characters = string.digits + string.ascii_uppercase
        # define training model

        input_tensor = Input((height, width, 3))

        x = input_tensor
        
        x = Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(x)
        x = MaxPooling2D((2, 2), name="pool1")(x)
        x = Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
        x =Reshape(target_shape=(5, 7680), name="reshape")(x)
        # x = BatchNormalization()(x)
        x = Dense(256, activation="relu", name="dense1")(x)
        x = Dense(64, activation="relu", name="dense2")(x)
        #x = Dense(36, activation="softmax", name="dense3")(x)
        #x = Activation('relu')(x)
        #x = MaxPooling2D(2)(x)


        x = Flatten()(x)
        x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
        self._model = Model(inputs=input_tensor, outputs=x)
        self.n_len = n_len
        self.n_class = n_class

    def _plot_model(self) -> None:
        plot_model(self._model, to_file="model.png", show_shapes=True)

    def load_model(self, model_path) -> None:
        self._model = load_model(model_path)

    def train(self, train_generator, test_generator) -> None:
        # self._plot_model()
        callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]

        self._model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(1e-4, amsgrad=True), 
                    metrics=['accuracy'],
                    steps_per_execution=1)
        self._model.fit(train_generator, epochs=1, validation_data=test_generator, workers=4, use_multiprocessing=True,
                            callbacks=callbacks,
                            steps_per_epoch = 2000)
    
    def predict(self, X) -> np.ndarray:
        if X.ndim == 3:  # (width, height, channel)
            np.expand_dims(X, axis=0)
        predict_prob = self._model.predict(X, verbose=0)
        predict_characters = self.decode(predict_prob)
        
        return np.array(predict_characters)

    def predicted_class(self, X) -> np.ndarray:
        if X.ndim == 3:  # (width, height, channel)
            np.expand_dims(X, axis=0)

        y = self._model.predict(X)
        y = np.array(y)    # y.shape = (digits in captcha, num of images, # classes)    
        y = np.resize(y, (y.shape[1],y.shape[0],y.shape[2])) # change dim 1 and dim 0
        y = np.argmax(np.array(y), axis=2)
        return np.array(y)

    def decode(self, y) -> np.ndarray:
        y = np.array(y)    # y.shape = (digits in captcha, num of images, # classes)
        predict_characters = []
        for i in range(0, y.shape[1]):
            single = np.argmax(np.array([y[:, i, :]]), axis=2)  # y.shape = (num of images, digits in captcha)
            captcha = ''.join(self.characters[z] for z in single[0])
            predict_characters.append(captcha)
        return np.array(predict_characters)
