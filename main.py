import keras
from setup import batch_size, epochs, num_classes, saveDir
from keras.datasets import cifar100
from utils import crop_image
from Autoencoder import Autoencoder


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