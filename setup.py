import os

os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)

batch_size = 256
num_classes = 10
epochs = 100
saveDir = "./pesos/"
train = False
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)