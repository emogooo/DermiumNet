CONVOLUTION_LAYERS = [3, 4, 5, 6, 7, 8, 9]
FILTERS = [16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256]
DENSE_LAYERS = [1, 2, 3, 4]
NEURONS = [16, 32, 64, 96, 112, 128, 144, 160, 176, 192, 256, 512]

CONVOLUTION_LAYERS_LEN_M1 = len(CONVOLUTION_LAYERS) - 1
FILTERS_LEN_M1 = len(FILTERS) - 1
DENSE_LAYERS_LEN_M1 = len(DENSE_LAYERS) - 1
NEURONS_LEN_M1 = len(NEURONS) - 1


class Hyperparameters:
    def __init__(self, convolutionIR: float, denseIR: float, filtersIR: list, neuronsIR: list) -> None:
        self.convolution = CONVOLUTION_LAYERS[round(convolutionIR)]
        self.dense = DENSE_LAYERS[round(denseIR)]
        self.filters = []
        for i in range(len(filtersIR)):
            self.filters.append(FILTERS[round(filtersIR[i])])

        self.neurons = []
        for i in range(len(neuronsIR)):
            self.neurons.append(NEURONS[round(neuronsIR[i])])

    def getHyperparameters(self):
        return f"Convolution Layer: {self.convolution}, Dense Layer: {self.dense}, Filters: {self.filters}, Neurons: {self.neurons}"
