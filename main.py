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

convolutionLayers = [3, 4, 5, 6, 7, 8, 9]
numberOfFilters = [16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256]  # Tekrarlanan
denseLayers = [1, 2, 3, 4]
numberOfNeurons = [16, 32, 64, 96, 112, 128, 144, 160, 176, 192, 256, 512]  # Tekrarlanan


print(convolutionLayers[-1])

print(random() * numberOfFilters[-1])  # bu doğru değil çünkü aralıklar çok büyük. 16-32 küçük, 256-512 çok büyük (eşit olması lazım)
print(numberOfFilters[round(random() * (len(numberOfFilters) - 1))])  # bu doğru, değişkende int değeri tutmayacaksın unutma! kesirli değeri tutacaksın ki değer değişebilsin. yoksa değişim oranı en az 1 olmalı ki parametre değişimi olsun.

maxConvolutionLayer = convolutionLayers[-1]
maxDenseLayer = denseLayers[-1]


class Agent:
    def __init__(self, conv: int, dense: int) -> None:
        self.convolutionLayer = conv
        self.denseLayer = dense
        self.convolutionLayerFilters = []
        self.denseLayerNeurons = []

        for i in range(conv):
            self.convolutionLayerFilters.append()

        for i in range(dense):
            self.denseLayerNeurons.append()


class Thesis:
    def __init__(self) -> None:
        self.CONVOLUTION_LAYERS = [3, 4, 5, 6, 7, 8, 9]
        self.NUMBER_OF_FILTERS = [16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256]  # Tekrarlanan
        self.DENSE_LAYERS = [1, 2, 3, 4]
        self.NUMBER_OF_NEURONS = [16, 32, 64, 96, 112, 128, 144, 160, 176, 192, 256, 512]  # Tekrarlanan

    def createFirstAgents(self, number: int):
        for i in range(number):
            pass


values = [5, 18, 32, 48, 10, 25, 60, 48, 0.0005]
hyperparameterList = [convolutionLayers, numberOfFilters, denseLayers, numberOfNeurons]


def find_closest_hyperparameters(values, hyperparameter_list):
    closest_list = []

    for value, hyperparameter in zip(values, hyperparameter_list):
        min_distance = float("inf")  # Positive infinity
        closest_hyperparameter = None

        for h in hyperparameter:
            distance = abs(value - h)
            if distance < min_distance:
                min_distance = distance
                closest_hyperparameter = h

        closest_list.append(closest_hyperparameter)

    return closest_list


print(find_closest_hyperparameters(values, hyperparameterList))
