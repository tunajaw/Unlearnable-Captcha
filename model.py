# ref: https://github.com/ypwhs/captcha_break

from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from IPython import display
from tqdm import tqdm
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
import numpy as np
import string

class model():
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
        for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
            for _ in range(n_cnn):
                x = Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            x = MaxPooling2D(2)(x)

        x = Flatten()(x)
        # x = Dropout(0.25)(x)
        x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
        self._model = Model(inputs=input_tensor, outputs=x)

    def _plot_model(self) -> None:
        plot_model(self._model, to_file="model.png", show_shapes=True)

    def load_model(self, model_path) -> None:
        self._model = load_model(model_path)

    def train(self, train_generator, test_generator) -> None:
        # self._plot_model()
        callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]

        self._model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(1e-3, amsgrad=True), 
                    metrics=['accuracy'])
        self._model.fit(train_generator, epochs=1, validation_data=test_generator, workers=4, use_multiprocessing=True,
                            callbacks=callbacks)
    
    def predict(self, X) -> string:

        if X.ndim == 3:  # only one image
            np.expand_dims(X, axis=0)

        predict_prob = self._model.predict(X)
        y = np.argmax(np.array(predict_prob), axis=2)
        print(y)
        predict_characters = ''.join([self.characters[z] for z in y])
        
        return predict_characters

    def decode(self, y) -> string:

        predict_characters = ''.join([self.characters[z] for z in y])
        return predict_characters
