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

from random import random, randint, choice

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

    def __init__(self, convolution: int, dense: int, filters: list, neurons: list) -> None:
        self.convolution = convolution
        self.dense = dense
        self.filters = filters
        self.neurons = neurons

    def __str__(self):
        roundedFilters = [round(x, 2) for x in self.filters]
        roundedNeurons = [round(x, 2) for x in self.neurons]
        return f"Convolution Layer: {round(self.convolution, 2)}, Dense Layer: {round(self.dense, 2)}, Filters: {roundedFilters}, Neurons: {roundedNeurons}"
    
    def setHyperparameterList():
        pass

class Thesis:
    def __init__(self) -> None:
        self.CONVOLUTION_LAYERS = [3, 4, 5, 6, 7, 8, 9]
        self.NUMBER_OF_FILTERS = [16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256]  # Tekrarlanan
        self.DENSE_LAYERS = [1, 2, 3, 4]
        self.NUMBER_OF_NEURONS = [16, 32, 64, 96, 112, 128, 144, 160, 176, 192, 256, 512]  # Tekrarlanan

    def createFirstAgents(self, number: int) -> list:
        agentList = []
        for _ in range(number):
            conv = random() * (len(self.CONVOLUTION_LAYERS) - 1)
            dense = random() * (len(self.DENSE_LAYERS) - 1)

            filters = []
            for _ in range(max(self.CONVOLUTION_LAYERS)):
                filters.append(random() * (len(self.NUMBER_OF_FILTERS) - 1))

            neurons = []
            for _ in range(max(self.DENSE_LAYERS)):
                neurons.append(random() * (len(self.NUMBER_OF_NEURONS) - 1))

            agent = Agent(conv, dense, filters, neurons)
            agentList.append(agent)

        return agentList

    def getHyperparameterList(self, reference: Agent) -> list:
        """
        Index değerlerini tam sayıya yuvarlayıp dizinin ilişkili elemanını alır, tüm elemanları aldıktan sonra dizi halinde döner.        
        """
        conv = self.CONVOLUTION_LAYERS[round(reference.convolution)]
        dense = self.DENSE_LAYERS[round(reference.dense)]
        filters = []
        for i in range(len(reference.filters)):
            filters.append(self.NUMBER_OF_FILTERS[round(reference.filters[i])])

        neurons = []
        for i in range(len(reference.neurons)):
            neurons.append(self.NUMBER_OF_NEURONS[round(reference.filters[i])])

        return [conv, dense, filters, neurons]


values = [5, 18, 32, 48, 10, 25, 60, 48, 0.0005]
hyperparameterList = [convolutionLayers, numberOfFilters, denseLayers, numberOfNeurons]



# print(find_closest_hyperparameters(values, hyperparameterList))

th = Thesis()
agents = th.createFirstAgents(5)
for agent in agents:
    realValue = th.getHyperparameterList(agent)
    print(realValue)
    print(agent)
    print()
