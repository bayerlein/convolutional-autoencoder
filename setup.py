import os

os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)

batch_size = 32
num_classes = 101
epochs = 20
saveDir = "./Testes/Teste2/"
train = False
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)