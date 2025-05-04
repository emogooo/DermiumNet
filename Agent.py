import Hyperparameters
from random import random


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

        if convolution > Hyperparameters.CONVOLUTION_LAYERS_LEN_M1:
            self.convolutionIndexReferences = Hyperparameters.CONVOLUTION_LAYERS_LEN_M1
        elif convolution < 0:
            self.convolutionIndexReferences = 0
        else:
            self.convolutionIndexReferences = convolution

        if dense > Hyperparameters.DENSE_LAYERS_LEN_M1:
            self.denseIndexReferences = Hyperparameters.DENSE_LAYERS_LEN_M1
        elif dense < 0:
            self.denseIndexReferences = 0
        else:
            self.denseIndexReferences = dense

        filterList = []
        for filter in filters:
            if filter > Hyperparameters.FILTERS_LEN_M1:
                filterList.append(Hyperparameters.FILTERS_LEN_M1)
            elif filter < 0:
                filterList.append(0)
            else:
                filterList.append(filter)

        self.filtersIndexReferences = filterList

        neuronList = []
        for neuron in neurons:
            if neuron > Hyperparameters.NEURONS_LEN_M1:
                neuronList.append(Hyperparameters.NEURONS_LEN_M1)
            elif neuron < 0:
                neuronList.append(0)
            else:
                neuronList.append(neuron)

        self.neuronsIndexReferences = neuronList

        self.hyperparameters = Hyperparameters.Hyperparameters(self.convolutionIndexReferences, self.denseIndexReferences, self.filtersIndexReferences, self.neuronsIndexReferences)

        self.number = Agent.counter

    def __str__(self) -> str:
        return f"Agent Number: {self.number}, Convolution Layer: {self.convolutionIndexReferences}, Dense Layer: {self.denseIndexReferences}, Filters: {self.filtersIndexReferences}, Neurons: {self.neuronsIndexReferences}"

    def getAgentValuesRounded(self) -> str:
        roundedFilters = [round(x, 2) for x in self.filtersIndexReferences]
        roundedNeurons = [round(x, 2) for x in self.neuronsIndexReferences]
        return f"Agent Number: {self.number}, Convolution Layer: {round(self.convolutionIndexReferences, 2)}, Dense Layer: {round(self.denseIndexReferences, 2)}, Filters: {roundedFilters}, Neurons: {roundedNeurons}"

    def getAgentParameters(self) -> list:
        return [self.number, self.convolutionIndexReferences, self.denseIndexReferences, self.filtersIndexReferences, self.neuronsIndexReferences]


def getFirstAgents(number: int) -> list[Agent]:
    agentList = []
    for _ in range(number):
        conv = random() * Hyperparameters.CONVOLUTION_LAYERS_LEN_M1
        dense = random() * Hyperparameters.DENSE_LAYERS_LEN_M1
        filters = []
        for _ in range(max(Hyperparameters.CONVOLUTION_LAYERS)):
            filters.append(random() * Hyperparameters.FILTERS_LEN_M1)

        neurons = []
        for _ in range(max(Hyperparameters.DENSE_LAYERS)):
            neurons.append(random() * Hyperparameters.NEURONS_LEN_M1)

        agent = Agent(conv, dense, filters, neurons)
        agentList.append(agent)

    return agentList