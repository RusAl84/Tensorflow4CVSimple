import matplotlib
import tensorflow

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers.core import Dense,Dropout

from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers import RMSprop,Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os



# Пути до файлов
dataset=r'D:\ML\train_img'
path_model=r"./models/simple_nn"
label_bin =r"./models/simple_nn.pickle"
plot= r"D:\ML\simple_nn_plot.png"

# инициализируем скорость обучения и общее число эпох
INIT_LR = 0.001
EPOCHS = 3

#INIT_LR = 0.0001
#EPOCHS = 1000



# инициализируем данные и метки
print("[INFO] Загрузка изображений...")
data = []
labels = []

# берём пути к изображениям и рандомно перемешиваем

imagePaths = sorted(list(paths.list_images(dataset)))
#print(imagePaths)
random.seed(42)
random.shuffle(imagePaths)


# цикл по изображениям
for imagePath in imagePaths:
	# загружаем изображение, меняем размер на 32x32 пикселей (без учета
	# соотношения сторон), сглаживаем его в 32x32x3=3072 пикселей и
	# добавляем в список
        image = cv2.imread(imagePath)
        try:
                image = cv2.resize(image, (16, 37)).flatten()
        except:
                continue
        data.append(image)

	# извлекаем метку класса из пути к изображению и обновляем
	# список меток
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# конвертируем метки из целых чисел в векторы (для 2х классов при
# бинарной классификации вам следует использовать функцию Keras
# “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
# не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# определим архитектуру 3072-1024-512-3 с помощью Keras
# 1776-1024-512-10
model = Sequential()
model.add(Dense(1024, input_shape=(1776,), activation="sigmoid"))
#model.add(Dense(2048, input_shape=(3072,), activation="sigmoid"))
model.add(Dropout(0.3)) #прореживание - увеличивать кол-во эпох при приминении, например до 200.
#model.add(Dense(1024, activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dropout(0.3))
model.add(Dense(258, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# компилируем модель, используя SGD как оптимизатор и категориальную
# кросс-энтропию в качестве функции потерь (для бинарной классификации
# следует использовать binary_crossentropy)
print("[INFO] training network...")

OPTIMIZER = SGD(lr=INIT_LR)
# OPTIMIZER = tensorflow.keras.optimizers.SGD(learning_rate=0.1)
#OPTIMIZER = Adam(lr=INIT_LR)


# model.compile(loss='mean_squared_error', optimizer='sgd')
# model.compile(loss="categorical_crossentropy",  optimizer=OPTIMIZER)
model.compile(loss="categorical_crossentropy",  optimizer=OPTIMIZER, metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,  metrics=["binary_accuracy"])

# обучаем нейросеть
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.plot(N, H.history["accuracy"], label="train_acc")
# plt.plot(N, H.history["val_accuracy"], label="val_acc")
#plt.plot(N, H.history["acc"], label="train_acc")
#plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot)

# сохраняем модель и бинаризатор меток на диск
print("[INFO] serializing network and label binarizer...")

model.save(path_model)
# model.save('model.model')
f = open(label_bin, "wb")
f.write(pickle.dumps(lb))
f.close()

#model.summary()
score = model.evaluate(testX, testY, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

