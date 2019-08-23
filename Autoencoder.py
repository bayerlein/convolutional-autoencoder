import keras
import setup
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, UpSampling2D,UpSampling3D, BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from setup import batch_size, epochs, kerasBKED, num_classes, saveDir

class Autoencoder:
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(lr=0.001)
        
        self.autoencoder = self.cria_modelo()
        self.carrega_pesos()

        self.autoencoder.compile(loss='mse', optimizer=optimizer)
        self.autoencoder.summary()
    def cria_modelo(self):
        entrada = Input(shape=self.img_shape)
        
        # encoder
        h = Conv2D(64, (5, 5), activation='relu', padding='same')(entrada)
        h = MaxPooling2D((2, 2), padding='same')(h)
        h = Conv2D(64, (5, 5), activation='relu', padding='same')(h)
        h = MaxPooling2D((2, 2), padding='same')(h)
        h = Dense(32)(h)
        
        # decoder
        h = UpSampling2D((2, 2))(h)
        h = Conv2D(64, (5, 5), activation='relu', padding='same')(h)
        h = UpSampling2D((2, 2))(h)
        saida = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(h)
        
        return Model(entrada, saida)
    def treina_modelo(self, x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size):
        caminho_diretorio = saveDir + 'AutoEncoder_pesos.hdf5'
        salva_pesos = ModelCheckpoint(filepath = caminho_diretorio, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
        parada = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1, 
                                       mode='auto')
        history = self.autoencoder.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[parada, salva_pesos])

    def prever(self, x_test):
        preds = self.autoencoder.predict(x_test)
        return preds
    
    def carrega_pesos(self):
        try:
            self.autoencoder.load_weights(saveDir + "AutoEncoder_pesos.hdf5")
            print(" ######## PESOS CARREGADOS ######## ")
        except Exception as e:
            print("NÃO EXISTE PESOS NA PASTA DEFINIDA\n")
            print(str(e))
            print(saveDir + "AutoEncoder_pesos.hdf5")
