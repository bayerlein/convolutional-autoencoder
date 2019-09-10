import tensorflow.keras
from setup import batch_size, epochs, num_classes, saveDir, train
from utils import crop_image, showOrigNoisy, showOrigNoisyRec, load_caltech101, plot_history
from Autoencoder import Autoencoder
import numpy as np

#load das imagens 
data_train, data_test = load_caltech101()

print('shapes train; test; val:')
print(data_train.shape)
print(data_test.shape)

# corrompe o centro da imagem
cropy = 40
cropx = 40
x_train_noisy = crop_image(data_train, cropx, cropy)
x_test_noisy = crop_image(data_test, cropx, cropy)
#x_val_noisy = crop_image(data_val, cropx, cropy)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#x_val_noisy = np.clip(x_val_noisy, 0., 1.)

showOrigNoisy(data_test, x_test_noisy, num=4)

ae = Autoencoder()

if train:
    ae.treina_modelo(x_train_noisy, data_train, x_test_noisy, data_test)
    print(ae.history.history.keys())
    plot_history(ae.history)


c10test = ae.prever(x_test_noisy)

#print("c10test: {0}\nc10val: {1}".format(np.average(c10test), np.average(c10val)))

showOrigNoisyRec(data_test, x_test_noisy, c10test)
