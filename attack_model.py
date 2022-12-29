import numpy as np
from keras import losses
import keras.backend as K
import sys
import tensorflow as tf


class attack_Model():
    def __init__(self, n_class, attack_method) -> None:
        # constants: must update when more attack methods are added into this repo.
        self.IMPLEMENTED_ATTACKS = ('FGSM')
        # variables
        self.epsilon = 4.0 / 256
        self.n_class = n_class
        self._attack_method = {}
        for a in attack_method:
            if(str(a) not in self.IMPLEMENTED_ATTACKS):
                raise NotImplementedError(f'{a} is not implemented. Available attack models: {self.IMPLEMENTED_ATTACKS}')
            else:
                # https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object
                self._attack_method[str(a)] = getattr(sys.modules[__name__], str(a))(self.epsilon, self.n_class)


    def test_single_attack_model(self, model_name, images, one_hot_label, proxy_model):
        model_name = str(model_name)
        if(model_name not in self._attack_method.keys()):
            raise ValueError(f'{model_name} is not in attack model. Your attack model uses {self._attack_method.keys()}.')
        else:
            return self._attack_method[model_name].generate_adversarial(images, one_hot_label, proxy_model)

    # Iteratively attack model
    # DO NOT CALL THIS FUNCTION AT SUBCLASS!!!
    def attack(self, model_name, images, labels, one_hot_labels, proxy_model, break_time=10):
        
        attacked_imgs = None
        # print(images.shape)
        # print(labels.shape)
        one_hot_labels = np.array(one_hot_labels)

        for i in range(images.shape[0]):
            print(i)
            _attacked = False
            _break = break_time
            attacked_img = np.array([images[i]])
            one_hot_label = np.array([one_hot_labels[:, i, :]]).reshape((one_hot_labels.shape[0], 1, one_hot_labels.shape[2]))
            label = [labels[i]]

            while((not _attacked) and _break):
                _break -= 1
                attacked_img = self.test_single_attack_model(model_name, attacked_img, one_hot_label, proxy_model)
                pred = proxy_model.predict(attacked_img)
                # check if every alphabet is break
                print(pred)
                if(sum([pred[0][j]!=label[0][j] for j in range(proxy_model.n_len)]) == proxy_model.n_len):
                    _attacked = True
            
            if(attacked_imgs is None):
                attacked_imgs = attacked_img
            else:
                attacked_imgs = np.vstack((attacked_imgs, attacked_img))

        print(attacked_imgs.shape)
        return attacked_imgs
        
        

class FGSM():
    def __init__(self, epsilon, n_class) -> None:
        # check if attack model is in parent model first
        self.epsilon = epsilon
        self.n_class = n_class
        self.sess = K.get_session()


    def generate_adversarial(self, images, labels, model):

        attacked_images = np.array([])


        with tf.GradientTape() as gtape:

            images = tf.convert_to_tensor(images)
            gtape.watch(images)

            preds = model._model(images)
            # print(pred)
            #loss = K.categorical_crossentropy(model._model.output, target)
            loss = losses.categorical_crossentropy(labels, preds)
            # print(loss)

        gradients = gtape.gradient(loss, images)#model._model.input)
        # print(gradients)
        sign = tf.sign(gradients)
        # print(sign)

        noise = self.epsilon * sign
        # print(noise)

        adversarial = tf.add(images, noise)
        # print(adversarial)

        adversarial_np = adversarial.numpy()
        adversarial_np = np.clip(adversarial_np, 0, 1)
        # print("adversarial_np")
        # print(adversarial_np.shape)
        attacked_images = np.append(attacked_images, adversarial_np)

        input_shape = np.array(images).shape
        attacked_images = attacked_images.reshape(input_shape)
        # print(np.array(images).shape)
        return attacked_images