import keras
from setup import batch_size, epochs, num_classes, saveDir, train
from keras.datasets import cifar100
from utils import crop_image, showOrigNoisy, showOrigNoisyRec
from Autoencoder import Autoencoder
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# normalizacao
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0])
print(x_test.shape[0])

# divide entre teste e validation
x_val = x_test[:7000]
x_test = x_test[7000:]

# corrompe o centro da imagem
cropy = 8
cropx = 8
x_train_noisy = crop_image(x_train, cropx, cropy)
x_test_noisy = crop_image(x_test, cropx, cropy)
x_val_noisy = crop_image(x_val, cropx, cropy)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
x_val_noisy = np.clip(x_val_noisy, 0., 1.)


#showOrigNoisy(x_train, x_train_noisy)

ae = Autoencoder()

ae.compile()
if train:
    ae.train(x_train_noisy, x_train, x_val_noisy, x_val)

score = ae.evaluate(x_test_noisy, x_test)
print(score)

c10test = ae.predict(x_test_noisy)
c10val = ae.predict(x_val_noisy)

print("c10test: {0}\nc10val: {1}".format(np.average(c10test), np.average(c10val)))

showOrigNoisyRec(x_test, x_test_noisy, c10test)
