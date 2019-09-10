import tensorflow.keras
import setup
import tensorflow.keras.layers as L
#from tensorflow.keras.layers import Reshape, Conv2DTranspose, Flatten, Input, Dense, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, UpSampling2D,UpSampling3D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from setup import batch_size, epochs, kerasBKED, num_classes, saveDir

class Autoencoder:
    def __init__(self):
        from tensorflow.keras.utils import plot_model
        self.img_rows = 144
        self.img_cols = 144
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(lr=0.001)
        
        encoder, decoder = self.cria_modelo()

        entrada = L.Input(self.img_shape)
        espaco_latente = encoder(entrada)
        reconstrucao = decoder(espaco_latente)

        self.autoencoder = tensorflow.keras.models.Model(inputs=entrada, outputs=reconstrucao)

        self.carrega_pesos()

        self.autoencoder.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        plot_model(self.autoencoder.layers[1], to_file='./Testes/Teste2/encoder.png', show_shapes=True, show_layer_names=True)
        plot_model(self.autoencoder.layers[2], to_file='./Testes/Teste2/decoder.png', show_shapes=True, show_layer_names=True)
    def cria_modelo(self):
        
        # encoder
        encoder = tensorflow.keras.models.Sequential()
        encoder.add(L.InputLayer(self.img_shape))

        encoder.add(L.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
        encoder.add(L.MaxPooling2D(pool_size=(2,2)))
        encoder.add(L.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
        encoder.add(L.MaxPooling2D(pool_size=(2,2)))
        encoder.add(L.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
        encoder.add(L.MaxPooling2D(pool_size=(2,2)))
        encoder.add(L.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
        encoder.add(L.MaxPooling2D(pool_size=(2,2)))
        encoder.add(L.Flatten())                  
        encoder.add(L.Dense(32))           

        # decoder
        decoder = tensorflow.keras.models.Sequential()
        decoder.add(L.InputLayer((32,)))
        decoder.add(L.Dense(9*9*256))
        decoder.add(L.Reshape((9,9,256)))
        decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))
        
        return encoder, decoder
    def treina_modelo(self, x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size):        
        caminho_diretorio = saveDir + 'AutoEncoder_pesos_caltech101_teste2.hdf5'
        salva_pesos = ModelCheckpoint(filepath = caminho_diretorio, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
        parada = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1, 
                                       mode='auto')
        self.history = self.autoencoder.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[parada, salva_pesos])

    def prever(self, x_test):
        preds = self.autoencoder.predict(x_test)
        return preds
    
    def carrega_pesos(self):
        try:
            self.autoencoder.load_weights(saveDir + "AutoEncoder_pesos_caltech101_teste2.hdf5")
            print(" ######## PESOS CARREGADOS ######## ")
        except Exception as e:
            print("NÃO EXISTE PESOS NA PASTA DEFINIDA\n")
            print(str(e))
            print(saveDir + "AutoEncoder_pesos.hdf5")
