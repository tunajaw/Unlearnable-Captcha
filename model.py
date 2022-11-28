# ref: https://github.com/ypwhs/captcha_break

from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from IPython import display
from tqdm import tqdm
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *

class Model():
    def __init__(self, height=64, width=128, n_len=4, n_class=36, model=None):
        '''
        CAPTCHA Break Model.

        height: height of CAPTCHA
        width: width of CAPTCHA
        n_len: sequence length of CAPTCHA
        n_class: how many possible characters in CAPTCHA
        model: which classification model to use
        '''
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
        x = Dropout(0.25)(x)
        x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
        self.model = Model(inputs=input_tensor, outputs=x)

    def _plot_model(self):
        plot_model(self.model, to_file="model.png", show_shapes=True)

    def train(self, train_generator, test_generator):
        callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]

        self.model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(1e-3, amsgrad=True), 
                    metrics=['accuracy'])
        self.model.fit_generator(train_generator, epochs=100, validation_data=test_generator, workers=4, use_multiprocessing=True,
                            callbacks=callbacks)
    
    def predict(self, X):
        return self.model.predict(X)