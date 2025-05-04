from random import random
from datetime import datetime
from time import sleep
import matplotlib.pyplot as plt
from Agent import *
import CNN
import visualkeras
import io
import json

iteration = 0
sleepTime = 0
startIndex = 0
aMultiplierEO = 0


def log(msg, path: str, mode: str, date: bool):
    if date:
        current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        print(current_time + " - " + msg)
        with open(path, mode) as file:
            file.write(f"{current_time} - {msg}")
    else:
        print(msg)
        with open(path, mode) as file:
            file.write(msg)


def readConfig(file_name="runtime-parameters.txt") -> dict:
    params = {}
    try:
        with open(file_name, "r") as file:
            for line in file:
                # Her satırı key: value şeklinde ayırma
                if line.strip():  # Boş satırları atla
                    key, value = line.strip().split(":")
                    params[key.strip()] = int(value.strip())
    except FileNotFoundError:
        print(f"{file_name} dosyası bulunamadı.")
    return params


def setConfig():
    global iteration, sleepTime, aMultiplierEO
    try:
        params = readConfig()
        iteration = params["iteration"]
        sleepTime = params["sleep"]
        aMultiplierEO = params["aMultiplierEO"] / 100.0
    except:
        pass


def readProgress(filename="progress.json"):
    try:
        with open(filename, "r") as file:
            progress_data = json.load(file)
            return progress_data
    except FileNotFoundError:
        print(f"{filename} dosyası bulunamadı, sıfırdan başlıyoruz.")
        return None
    except json.JSONDecodeError:
        print(f"{filename} dosyasında hata var, sıfırdan başlıyoruz.")
        return None


def saveProgress(index, agents, globalBestAgent, globalBestFitness, localFitnessChart, globalFitnessChart, filename="progress.json"):
    progress_data = {"index": index, "agents": agents, "globalBestAgent": globalBestAgent, "globalBestFitness": globalBestFitness, "localFitnessChart": localFitnessChart, "globalFitnessChart": globalFitnessChart}

    # Veriyi JSON formatında kaydet
    try:
        with open(filename, "w") as file:
            json.dump(progress_data, file)
        print("İlerleme kaydedildi.")
    except Exception as e:
        print(f"İlerleme kaydedilirken hata oluştu: {e}")


numberOfAgents = 5
agents = getFirstAgents(numberOfAgents)
globalBestAgent = agents[0]
globalBestFitness = 0
localBestAgent = None
localBestFitness = 0
localFitnessChart = []
globalFitnessChart = []

setConfig()

try:
    progress = readProgress()
    if progress != None:
        startIndex = progress["index"]
        GB_AgentParameters = progress["globalBestAgent"]
        globalBestFitness = progress["globalBestFitness"]
        localFitnessChart = progress["localFitnessChart"]
        globalFitnessChart = progress["globalFitnessChart"]
        agentsParameters = progress["agents"]
        newAgentList = []
        for ap in agentsParameters:
            agent = Agent(ap[1], ap[2], ap[3], ap[4])
            agent.number = ap[0]
            newAgentList.append(agent)

        globalBestAgent = Agent(GB_AgentParameters[1], GB_AgentParameters[2], GB_AgentParameters[3], GB_AgentParameters[4])
        globalBestAgent.number = GB_AgentParameters[0]

        Agent.counter = agentsParameters[0][0] + 4

        agents = newAgentList
        print("Veriler okundu kalınan yerden devam ediliyor.")

except:
    print("Veriler okunmaya çalışırken bir hata oluştu.")


for iter in range(startIndex, iteration):

    setConfig()

    log(f"\n---------------------------- ITERASYON : {iter + 1} ----------------------------\n", "results/System-Log.txt", "a", True)
    localBestFitness = 0
    for agent in agents:
        log(f"Iteration: {iter + 1} - {agent}\n", "results/Hyperparameter-Log.txt", "a", True)
        model = CNN.CNN.getModel(agent)
        result = CNN.CNN.getModelResult(agent.number, model)
        accuracy = result.getAccuracyScore(3)
        log(f"Agent Number: {agent.number}, accuracy: {accuracy}, {agent.hyperparameters.getHyperparameters()}\n", "results/System-Log.txt", "a", True)

        if accuracy > localBestFitness:
            localBestFitness = accuracy
            localBestAgent = agent

        if accuracy > globalBestFitness:  # neden accuracy ile başarı oranını tespit ediyoruz? konfüzyon matrisinden çıkarılan bazı değerleri kıstas almak daha mantıklı değil mi?
            globalBestFitness = accuracy
            globalBestAgent = agent
            result.saveConfusionMatrixChart()
            result.saveAccuracyChart()
            result.saveLossChart()
            log(f"Daha iyi bir agent bulundu. Accuracy oranı : {globalBestFitness}, agent no: {globalBestAgent.number}\n", "results/System-Log.txt", "a", True)

        sleep(sleepTime)

    log(f"Iteration End: {iter+1}\n", "results/Hyperparameter-Log.txt", "a", True)
    log(f"{iter+1}. setin en iyi accuracy değeri: {localBestFitness}, en iyi agent no: {localBestAgent.number}\n", "results/System-Log.txt", "a", True)
    log(f"{round(localBestFitness, 3)},", "results/Local_Best_Fitness.txt", "a", False)
    log(f"{round(globalBestFitness, 3)},", "results/Global_Best_Fitness.txt", "a", False)
    localFitnessChart.append(localBestFitness)
    globalFitnessChart.append(globalBestFitness)
    log(f"Local best fitness array: {localFitnessChart}\n", "results/System-Log.txt", "a", True)
    log(f"Global best fitness array: {globalFitnessChart}\n", "results/System-Log.txt", "a", True)

    newAgentList = []
    for i, agent in enumerate(agents):
        a = aMultiplierEO * (1 - iter / iteration)
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
        log(f"Yeni hiperparametre değerleri (sınır bağımsız) oluşturuldu: Agent Number: {newAgent.number}, Convolution Layer: {round(newConv, 2)}, Dense Layer: {round(newDense, 2)}, Filters: {roundedFilters}, Neurons: {roundedNeurons}\n", "results/System-Log.txt", "a", True)
        log(f"Yeni agent oluşturuldu------------------------------------: {newAgent.getAgentValuesRounded()}\n", "results/System-Log.txt", "a", True)

    agents = newAgentList

    agentParameterList = []
    for agent in newAgentList:
        agentParameterList.append(agent.getAgentParameters())
    saveProgress(index=iter + 1, agents=agentParameterList, globalBestAgent=globalBestAgent.getAgentParameters(), globalBestFitness=globalBestFitness, localFitnessChart=localFitnessChart, globalFitnessChart=globalFitnessChart)

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
