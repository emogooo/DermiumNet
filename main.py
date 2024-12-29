from random import random
from datetime import datetime
import matplotlib.pyplot as plt
from Agent import Agent, getFirstAgents
import CNN
import visualkeras
import io

ITERATION = 100


def logAgent(iter: int, agent: Agent):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results/Hyperparameter-Log.txt", "a") as file:
        file.write(f"{current_time} - iter: {iter} - {agent}\n")


def logMessage(msg: str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time + " - " + msg)
    with open("results/System-Log.txt", "a") as file:
        file.write(f"{current_time} - {msg}\n")


numberOfAgents = 5
agents = getFirstAgents(numberOfAgents)
globalBestAgent = agents[0]
globalBestFitness = 0
localBestAgent = None
localBestFitness = 0
localFitnessChart = []
globalFitnessChart = []

for iter in range(ITERATION):

    logMessage(f"iter: {iter}")
    localBestFitness = 0
    for agent in agents:
        logAgent(iter, agent)
        model = CNN.CNN.getModel(agent)
        result = CNN.CNN.getModelResult(agent.number, model)
        accuracy = result.getAccuracyScore(3)
        logMessage(f"Agent Number: {agent.number}, accuracy: {accuracy}, " + agent.hyperparameters.getHyperparameters())

        if accuracy > localBestFitness:
            localBestFitness = accuracy
            localBestAgent = agent

        if accuracy > globalBestFitness:  # neden accuracy ile başarı oranını tespit ediyoruz? konfüzyon matrisinden çıkarılan bazı değerleri kıstas almak daha mantıklı değil mi?
            globalBestFitness = accuracy
            globalBestAgent = agent
            result.saveConfusionMatrixChart()
            result.saveAccuracyChart()
            result.saveLossChart()
            logMessage(f"Daha iyi bir model bulundu. Doğruluk oranı : {globalBestFitness}, model no: {globalBestAgent.number}")

    logMessage(f"{iter}. setin en iyi agent'ının fitness değeri: {localBestFitness}, en iyi agent no: {localBestAgent.number}")
    localFitnessChart.append(localBestFitness)
    globalFitnessChart.append(globalBestFitness)

    newAgentList = []
    for i, agent in enumerate(agents):
        a = 0.2 * (1 - iter / ITERATION)
        r1 = random()
        r2 = random()
        g = random()

        gMultiplier = 1
        if g < 0.5:
            gMultiplier = 1
        else:
            gMultiplier = -1

        newConv = agent.convolutionIndexReferences + (gMultiplier * a * r1 * (globalBestAgent.convolutionIndexReferences - (r2 * agent.convolutionIndexReferences)))
        newDense = agent.denseIndexReferences + (gMultiplier * a * r1 * (globalBestAgent.denseIndexReferences - (r2 * agent.denseIndexReferences)))

        newFilters = []
        for i in range(len(agent.filtersIndexReferences)):
            x = agent.filtersIndexReferences[i] + (gMultiplier * a * r1 * (globalBestAgent.filtersIndexReferences[i] - (r2 * agent.filtersIndexReferences[i])))
            newFilters.append(x)

        newNeurons = []
        for i in range(len(agent.neuronsIndexReferences)):
            x = agent.neuronsIndexReferences[i] + (gMultiplier * a * r1 * (globalBestAgent.neuronsIndexReferences[i] - (r2 * agent.neuronsIndexReferences[i])))
            newNeurons.append(x)

        newAgent = Agent(newConv, newDense, newFilters, newNeurons)
        newAgentList.append(newAgent)
        roundedFilters = [round(x, 2) for x in newFilters]
        roundedNeurons = [round(x, 2) for x in newNeurons]
        logMessage(f"Yeni hiperparametre değerleri (sınır bağımsız) oluşturuldu: Agent Number: {newAgent.number}, Convolution Layer: {round(newConv, 2)}, Dense Layer: {round(newDense, 2)}, Filters: {roundedFilters}, Neurons: {roundedNeurons}")
        logMessage(f"Yeni agent oluşturuldu------------------------------------: {newAgent.getAgentValuesRounded()}")

    agents = newAgentList
    logMessage(f"Local best fitness log: {localFitnessChart}")
    logMessage(f"Global best fitness log: {globalFitnessChart}")

iterations = list(range(1, len(localFitnessChart) + 1))

plt.figure(figsize=(10, 5))
plt.plot(iterations, localFitnessChart, marker="o", label="Local Fitness", linestyle="-")
plt.title("Local Fitness Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Fitness Value")
plt.grid()
plt.legend()
plt.savefig(f"{CNN.DIRECTORY}/results/local_fitness")

plt.figure(figsize=(10, 5))
plt.plot(iterations, globalFitnessChart, label="Global Fitness", color="orange", linestyle="-")
plt.title("Global Fitness Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Fitness Value")
plt.grid()
plt.legend()
plt.savefig(f"{CNN.DIRECTORY}/results/global_fitness")

model = CNN.CNN.getModel(globalBestAgent)
model.save(f"{CNN.DIRECTORY}/results/best_model.h5")

# from tensorflow.keras.models import load_model
# loaded_model = load_model(f"{CNN.DIRECTORY}/results/best_model.h5")
visualkeras.layered_view(model, to_file=f"{CNN.DIRECTORY}/results/best_model_architecture.png", legend=True, scale_xy=0.8, scale_z=0.1, spacing=10)  # Katman türlerini gösteren bir açıklama ekler  # Grafik boyutunu küçültür  # Z eksenini küçültür  # Katmanlar arasındaki boşluğu ayarlar

model_summary = io.StringIO()
model.summary(print_fn=lambda x: model_summary.write(x + "\n"))
summary_text = model_summary.getvalue()

with open(f"{CNN.DIRECTORY}/results/best_model_summary.txt", "w") as file:
    file.write(summary_text)
