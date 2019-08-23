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
        plt.imshow(orig[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display original
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(noise[i].reshape(32, 32, 3))
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
        plt.imshow(orig[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # corrompida
        ax = plt.subplot(3, n, i +1 + n)
        plt.imshow(noise[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # recuperada
        ax = plt.subplot(3, n, i +1 + n + n)
        plt.imshow(denoise[i].reshape(32, 32, 3))
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
    import keras.backend as K    

    _ssim = tf.image.ssim(y_true, y_pred, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return tf.reduce_mean((1 - _ssim)/2)