# Çiçek datasetini kanser datasetine benzeterek ayarlamaya çalış. 80-10-10 oranı iyi olabilir.
# İlk olarak hiperparametre havuzu ve dönüş metodu oluştur.
# Ardından EO algoritmasını entegre et.
# Örnek modeli model.pyden al.
# Çalışma testlerine başla.
# Görsellik çok önemli o kısma ayrıca çalışmalıyız.

"""
BU İKİ YÖNTEM DE DENENECEK.

# filterelerin hepsini oluştur ve tut.
# katman sayısı kadarını kullan?

# katman sayısını oluştur,
# katman sayısı kadar filtre oluştur.


ilk olarak sade bir hiperparametre listesi ile başlıyorum, ileride bu listeye blockLayers, blockDepth ve layerHierarchy listelerini de ekleyeceğim.
blockLayers = [0, 1, 2, 3, 4]
blockDepth = [2, 3]  # Tekrarlanan
"""

from random import random
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.backend import clear_session
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

convolutionLayers = [3, 4, 5, 6, 7, 8, 9]
numberOfFilters = [16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256]  # Tekrarlanan
denseLayers = [1, 2, 3, 4]
numberOfNeurons = [16, 32, 64, 96, 112, 128, 144, 160, 176, 192, 256, 512]  # Tekrarlanan


# print(convolutionLayers[-1])

# print(random() * numberOfFilters[-1])  # bu doğru değil çünkü aralıklar çok büyük. 16-32 küçük, 256-512 çok büyük (eşit olması lazım)
# print(numberOfFilters[round(random() * (len(numberOfFilters) - 1))])  # bu doğru, değişkende int değeri tutmayacaksın unutma! kesirli değeri tutacaksın ki değer değişebilsin. yoksa değişim oranı en az 1 olmalı ki parametre değişimi olsun.

maxConvolutionLayer = convolutionLayers[-1]
maxDenseLayer = denseLayers[-1]


class Agent:
    """
    Tüm parametre değerleri float olarak tutulmalı. Tutulan değerler ilgili dizinin index değerleridir. Gerçek değerler değil. Bu şekilde tutulma amacı ise, evrimsel algoritmanın her optimizasyonunda değer ufak ufak değişeceği için önceki değişiklikleri
    kaybetmememiz lazım. Ya da değişikliklerin 1'den büyük olması lazım ki her optimizasyonda farklı bir hiperparametre listesi oluşsun (Tabii ki bu adımları çok büyük tutacağı için local minimumu atlamak anlamına gelir,
    dolayısıyla bu çözüm yöntemi mantıklı değildir). Her optimizasyonda değerlerimiz azar azar değişecektir. Eğer biz bu değişiklikleri her optimizasyonda 16,32,64,96,128 gibi liste değerlerine ayarlarsak evrimsel algoritma sürekli aynı
    hiperparametre listesini üretir. Çünkü değişim oranı, bu sınırlar içerisinde kalabilir. Zaman içerisinde değişim istiyorsak önceki değişikliklerin kaydedilmesi şarttır. İşte bu yüzden getHyperparameterList isimli bir metodumuz mevcuttur.
    """

    counter = 0

    def __init__(self, convolution: float, dense: float, filters: list, neurons: list) -> None:
        Agent.counter += 1
        self.convolution = convolution
        self.dense = dense
        self.filters = filters
        self.neurons = neurons
        self.number = Agent.counter

    def __str__(self):
        # roundedFilters = [round(x, 2) for x in self.filters]
        # roundedNeurons = [round(x, 2) for x in self.neurons]
        # return f"Convolution Layer: {round(self.convolution, 2)}, Dense Layer: {round(self.dense, 2)}, Filters: {roundedFilters}, Neurons: {roundedNeurons}"
        return f"Agent Number: {self.number}, Convolution Layer: {self.convolution}, Dense Layer: {self.dense}, Filters: {self.filters}, Neurons: {self.neurons}"


class ModelResult:

    def __init__(self, agentNumber, history, predictions, trueClasses, classLabels) -> None:
        self.__HISTORY = history
        self.__PRETICTED_CLASSES = np.argmax(predictions, axis=1)
        self.__TRUE_CLASSES = trueClasses
        self.__CLASS_LABELS = classLabels
        self.__MODEL_NUMBER = agentNumber

    def __getConfusionMatrix(self):
        return confusion_matrix(self.__TRUE_CLASSES, self.__PRETICTED_CLASSES)

    def getAccuracyScore(self, fp: int) -> float:
        if fp == 0:
            return round(accuracy_score(self.__TRUE_CLASSES, self.__PRETICTED_CLASSES))
        return round(accuracy_score(self.__TRUE_CLASSES, self.__PRETICTED_CLASSES), fp)

    def saveConfusionMatrixChart(self, directory: str):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.__getConfusionMatrix(), annot=True, fmt="d", cmap="Blues", xticklabels=self.__CLASS_LABELS, yticklabels=self.__CLASS_LABELS)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix\nModel: {self.__MODEL_NUMBER}\nTotal Accuracy: {self.getAccuracyScore(2)}")
        plt.savefig(f"{directory}/results/confusion_matrix/model_{self.__MODEL_NUMBER}_acc_{self.getAccuracyScore(0)}")

    def saveLossChart(self, directory: str):
        _, ax = plt.subplots()
        ax.set_xlabel("Epoch", loc="right")
        plt.title(f"Loss - Validation Loss\nModel: {self.__MODEL_NUMBER}")
        plt.plot(self.__HISTORY.history["loss"], "green", label="Loss")
        plt.plot(self.__HISTORY.history["val_loss"], "purple", label="Validation Loss")
        plt.legend()
        plt.savefig(f"{directory}/results/loss/model_{self.__MODEL_NUMBER}")

    def saveAccuracyChart(self, directory: str):
        _, ax = plt.subplots()
        ax.set_xlabel("Epoch", loc="right")
        plt.title(f"Accuracy - Validation Accuracy\nModel: {self.__MODEL_NUMBER}")
        plt.plot(self.__HISTORY.history["accuracy"], "red", label="Accuracy")
        plt.plot(self.__HISTORY.history["val_accuracy"], "blue", label="Validation Accuracy")
        plt.legend()
        plt.savefig(f"{directory}/results/accuracy/model_{self.__MODEL_NUMBER}")


class Thesis:

    def __init__(self, directory: str) -> None:
        self.CONVOLUTION_LAYERS = [3, 4, 5, 6, 7, 8, 9]
        self.FILTERS = [16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256]  # Tekrarlanan
        self.DENSE_LAYERS = [1, 2, 3, 4]
        self.NEURONS = [16, 32, 64, 96, 112, 128, 144, 160, 176, 192, 256, 512]  # Tekrarlanan

        self.INPUT_SHAPE = (224, 224, 3)
        self.BATCH_SIZE = 16
        self.EPOCH = 10
        self.ITERATION = 100

        self.directory = directory

        trainDatagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        testDatagen = ImageDataGenerator(rescale=1.0 / 255)

        datasetDirectory = directory + "/datasets/flower/modelCrashDebugSet/"  # "/datasets/flowersTestAugmentSplit/"

        self.trainingSet = trainDatagen.flow_from_directory(datasetDirectory + "train", batch_size=self.BATCH_SIZE, class_mode="categorical")
        self.validationSet = testDatagen.flow_from_directory(datasetDirectory + "validation", batch_size=self.BATCH_SIZE, class_mode="categorical")
        self.testSet = testDatagen.flow_from_directory(datasetDirectory + "test", batch_size=1, class_mode="categorical", shuffle=False)

        self.CLASS_LABELS = list(self.testSet.class_indices.keys())
        self.TRUE_CLASSES = self.testSet.classes

    def getFirstAgents(self, number: int) -> list[Agent]:
        agentList = []
        for _ in range(number):
            conv = random() * (len(self.CONVOLUTION_LAYERS) - 1)
            dense = random() * (len(self.DENSE_LAYERS) - 1)

            filters = []
            for _ in range(max(self.CONVOLUTION_LAYERS)):
                filters.append(random() * (len(self.FILTERS) - 1))

            neurons = []
            for _ in range(max(self.DENSE_LAYERS)):
                neurons.append(random() * (len(self.NEURONS) - 1))

            agent = Agent(conv, dense, filters, neurons)
            agentList.append(agent)

        return agentList

    def getHyperparameterList(self, agent: Agent) -> str:
        conv = self.CONVOLUTION_LAYERS[round(agent.convolution)]
        dense = self.DENSE_LAYERS[round(agent.dense)]
        filters = []
        for i in range(len(agent.filters)):
            filters.append(self.FILTERS[round(agent.filters[i])])

        neurons = []
        for i in range(len(agent.neurons)):
            neurons.append(self.NEURONS[round(agent.neurons[i])])

        return f"Convolution Layer: {conv}, Dense Layer: {dense}, Filters: {filters}, Neurons: {neurons}"

    def getModel(self, agent: Agent) -> Sequential:
        model = Sequential()
        for i in range(self.CONVOLUTION_LAYERS[round(agent.convolution)]):
            if i == 0:
                model.add(Conv2D(filters=self.FILTERS[round(agent.filters[i])], kernel_size=(3, 3), activation="relu", padding="same", input_shape=self.INPUT_SHAPE))
            else:
                model.add(Conv2D(filters=self.FILTERS[round(agent.filters[i])], kernel_size=(3, 3), activation="relu", padding="same"))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(GlobalAveragePooling2D())

        for i in range(self.DENSE_LAYERS[round(agent.dense)]):
            model.add(Dense(units=self.NEURONS[round(agent.neurons[i])], activation="relu"))
            model.add(Dropout(0.25))

        model.add(Dense(units=5, activation="softmax"))
        return model

    def getModelResult(self, agentNumber, model: Sequential) -> ModelResult:
        model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        reduceLearningRate = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=1e-4, verbose=1)
        checkpoint = ModelCheckpoint(filepath="results/model_checkpoint.h5", monitor="val_loss", save_best_only=True, verbose=1)
        history = model.fit(self.trainingSet, steps_per_epoch=self.trainingSet.samples // self.BATCH_SIZE, epochs=self.EPOCH, validation_data=self.validationSet, validation_steps=self.validationSet.n // self.validationSet.batch_size, callbacks=[reduceLearningRate, checkpoint])
        predictions = model.predict(self.testSet, steps=self.testSet.samples // self.testSet.batch_size)
        result = ModelResult(agentNumber, history, predictions, self.TRUE_CLASSES, self.CLASS_LABELS)
        clear_session()
        tf.keras.backend.clear_session()
        return result

    def logAgent(self, iter: int, agent: Agent):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("results/Hyperparameter-Log.txt", "a") as file:
            file.write(f"{current_time} - iter: {iter} - {agent}\n")

    def logMessage(self, msg: str):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time + " - " + msg)
        with open("results/System-Log.txt", "a") as file:
            file.write(f"{current_time} - {msg}\n")

    def equilibriumOptimizer(self):
        numberOfAgents = 5
        agents = self.getFirstAgents(numberOfAgents)
        globalBestAgent = agents[0]
        globalBestFitness = float("inf")
        fitnessChart = []
        localBestAgent = None
        localBestFitness = 0

        for iter in range(self.ITERATION):

            self.logMessage(f"iter: {iter}")

            for agent in agents:
                self.logAgent(iter, agent)
                self.logMessage(f"Agent Number: {agent.number}, " + self.getHyperparameterList(agent))
                try:
                    model = self.getModel(agent)
                except Exception as e:
                    self.logMessage(e)
                    continue

                result = self.getModelResult(agent.number, model)
                accuracy = result.getAccuracyScore(3)

                if accuracy > localBestFitness:
                    localBestFitness = accuracy
                    localBestAgent = agent

                if accuracy > globalBestFitness:  # neden accuracy ile başarı oranını tespit ediyoruz? konfüzyon matriksindeki bazı değerleri kıstas almak daha mantıklı değil mi?
                    globalBestFitness = accuracy
                    globalBestAgent = agent
                    result.saveConfusionMatrixChart(self.directory)
                    result.saveAccuracyChart(self.directory)
                    result.saveLossChart(self.directory)
                    self.logMessage(f"Daha iyi bir model bulundu. Doğruluk oranı : {globalBestFitness}, iterasyon = {iter}")

            self.logMessage(f"{iter}. setin en iyi agent'ının fitness değeri: {localBestFitness}, en iyi agent no: {localBestAgent.number}")
            fitnessChart.append(localBestFitness)

            for i, agent in enumerate(agents):
                a = 2 * (1 - iter / self.ITERATION)
                r1 = random()
                r2 = random()
                g = random()

                newConv = None
                newDense = None
                newFilters = []
                newNeurons = []

                if g < 0.5:
                    newConv = agent.convolution + a * r1 * (globalBestAgent.convolution - r2 * agent.convolution)
                    newDense = agent.dense + a * r1 * (globalBestAgent.dense - r2 * agent.dense)

                    for i in range(len(agent.filters)):
                        x = agent.filters[i] + a * r1 * (globalBestAgent.filters[i] - r2 * agent.filters[i])
                        newFilters.append(x)

                    for i in range(len(agent.neurons)):
                        x = agent.neurons[i] + a * r1 * (globalBestAgent.neurons[i] - r2 * agent.neurons[i])
                        newNeurons.append(x)
                else:
                    newConv = agent.convolution - a * r1 * (globalBestAgent.convolution - r2 * agent.convolution)
                    newDense = agent.dense - a * r1 * (globalBestAgent.dense - r2 * agent.dense)

                    for i in range(len(agent.filters)):
                        x = agent.filters[i] - a * r1 * (globalBestAgent.filters[i] - r2 * agent.filters[i])
                        newFilters.append(x)

                    for i in range(len(agent.neurons)):
                        x = agent.neurons[i] - a * r1 * (globalBestAgent.neurons[i] - r2 * agent.neurons[i])
                        newNeurons.append(x)

                agents[i] = Agent(newConv, newDense, newFilters, newNeurons)
                self.logMessage(f"Yeni agent oluşturuldu: {agents[i]}")


th = Thesis("D:/Github/DermiumNet")
th.equilibriumOptimizer()
