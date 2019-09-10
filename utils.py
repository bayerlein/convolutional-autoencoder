# faz um recorte com tamanho cropx e cropy em uma imagem
def crop_image(img_np_array,cropx,cropy):
    import numpy as np
    img_np_array_copy = np.copy(img_np_array)
    for img in img_np_array_copy:
      
      y, x, z = img.shape
      startx = x//2-(cropx//2)
      starty = y//2-(cropy//2)    
      img[starty:starty+cropy,startx:startx+cropx] = 0
    return img_np_array_copy

# exibe imagens
def showOrigNoisy(orig, noise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig[i].reshape(144, 144, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display noise
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(noise[i].reshape(144, 144, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# exibe imagens
def showOrigNoisyRec(orig, noise, denoise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(orig[i].reshape(144, 144, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # corrompida
        ax = plt.subplot(3, n, i +1 + n)
        plt.imshow(noise[i].reshape(144, 144, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # recuperada
        ax = plt.subplot(3, n, i +1 + n + n)
        plt.imshow(denoise[i].reshape(144, 144, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# funcao de similiaridade
# TODO - validar como usar no modelo
'''
    Funcao compara e retorna a semelhança entre duas imagens (y_true e y_pred)
'''
def ssim_loss(y_true, y_pred):
    import tensorflow as tf
    from skimage.measure import compare_ssim as ssim
    import tensorflow.keras.backend as K    

    _ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03, power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    return tf.reduce_mean(1 - _ssim)

'''
    Funcao processa o dataset e retorna dados para test, validation e trainamento
'''
def load_caltech101(path = "./dataset_caltech101/"):
    print("carregando imagens")
    import cv2
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    categories = sorted(os.listdir(path))
    data = []

    for i, category in enumerate(categories):
        for f in os.listdir(path + "/" + category):
            ext = os.path.splitext(f)[1]
            fullpath = os.path.join(path + "/" + category, f)
            #print(fullpath)
            label = fullpath.split(os.path.sep)[-2]
            image = cv2.imread(fullpath)
            image = cv2.resize(image, (144, 144))
            data.append(image / 255)
        
    #data = np.array(data, dtype="float")# / 255.0
    #data = data / 255
    print(len(data))

    data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)

    data_train = np.array(data_train, dtype="float")

    data_test = np.array(data_test, dtype="float")

    #print(data_train.shape)
    return data_train, data_test

def plot_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Testes/Teste2/model_accuracy.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Testes/Teste2/model_loss.png')
    plt.show()
