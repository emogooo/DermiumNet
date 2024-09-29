# +++++ Çiçek datasetini kanser datasetine benzeterek ayarlamaya çalış. 80-10-10 oranı iyi olabilir.
# İlk olarak hiperparametre havuzu oluştur.
# Ardından EO algoritmasını entegre et.
# Örnek modeli model.pyden al.
# Çalışma testlerine başla.
# Görsellik çok önemli o kısma ayrıca çalışmalıyız.

import numpy as np

# Parameters and their ranges as arrays
convolution_layers = np.array([3, 4, 5, 6, 7, 8, 9, 10])
number_of_filters = np.array([16, 32, 48, 64, 96, 128, 144, 160, 176, 192, 256])
optimization_algorithms = np.array(['Adam', 'SGD with Nesterov', 'Nadam'])
learning_rate = np.array([0.0001, 0.0005, 0.001])
neurons_dense1 = np.array([32, 64, 96, 128, 144, 160, 176, 192, 256])
neurons_dense2 = np.array([0, 16, 32, 64, 96, 112, 128])
