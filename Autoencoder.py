import keras
import setup
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, UpSampling2D,UpSampling3D, BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from setup import batch_size, epochs, kerasBKED, num_classes, saveDir

class Autoencoder:
    def __init__(self):
        self.input_img = Input(shape=(32, 32, 3))
        x = Dropout(0.5)
        x = Conv2D(64, (3, 3), padding='same')(self.input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        self.encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Dropout(0.5)
        x = Conv2D(16, (3, 3), padding='same')(self.encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        self.decoded = Activation('sigmoid')(x)

        self.model = Model(self.input_img, self.decoded)

        # carrega pesos
        try:
            self.model.load_weights(saveDir + "AutoEncoder_pesos.hdf5")
            print(" ######## PESOS CARREGADOS ######## ")
        except:
            print("NÃO EXISTE PESOS NA PASTA DEFINIDA\n")
            print(saveDir + "AutoEncoder_pesos.hdf5")

    def compile(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self, x_train, target, x_val, target_val):
        es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        chkpt = saveDir + 'AutoEncoder_pesos.hdf5'
        cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        self.history = self.model.fit(x_train, target,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, target_val),
                    callbacks=[es_cb, cp_cb],
                    shuffle=True)
        return self.history

    def evaluate(self, x_test, x_test_target):
        score = self.model.evaluate(x_test, x_test_target, verbose=1)
        return score

    def predict(self, test):
        return self.model.predict(test)
