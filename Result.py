from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Result:

    def __init__(self, dir, agentNumber, history, predictions, trueClasses, classLabels) -> None:
        self.__DIRECTORY = dir
        self.__HISTORY = history
        self.__PRETICTED_CLASSES = np.argmax(predictions, axis=1)
        self.__TRUE_CLASSES = trueClasses
        self.__CLASS_LABELS = classLabels
        self.__MODEL_NUMBER = agentNumber

    def __getConfusionMatrix(self):
        return confusion_matrix(self.__TRUE_CLASSES, self.__PRETICTED_CLASSES)

    def getAccuracyScore(self, fp: int = -1) -> float:
        if fp == -1:
            return str(round(accuracy_score(self.__TRUE_CLASSES, self.__PRETICTED_CLASSES), 2)).replace(".", "_")
        return round(accuracy_score(self.__TRUE_CLASSES, self.__PRETICTED_CLASSES), fp)

    def saveConfusionMatrixChart(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.__getConfusionMatrix(), annot=True, fmt="d", cmap="Blues", xticklabels=self.__CLASS_LABELS, yticklabels=self.__CLASS_LABELS)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix\nModel: {self.__MODEL_NUMBER}\nTotal Accuracy: {self.getAccuracyScore(2)}")
        plt.savefig(f"{self.__DIRECTORY}/results/confusion_matrix/model_{self.__MODEL_NUMBER}_acc_{self.getAccuracyScore()}")

    def saveLossChart(self):
        _, ax = plt.subplots()
        ax.set_xlabel("Epoch", loc="right")
        plt.title(f"Loss - Validation Loss\nModel: {self.__MODEL_NUMBER}")
        plt.plot(self.__HISTORY.history["loss"], "green", label="Loss")
        plt.plot(self.__HISTORY.history["val_loss"], "purple", label="Validation Loss")
        plt.legend()
        plt.savefig(f"{self.__DIRECTORY}/results/loss/model_{self.__MODEL_NUMBER}")

    def saveAccuracyChart(self):
        _, ax = plt.subplots()
        ax.set_xlabel("Epoch", loc="right")
        plt.title(f"Accuracy - Validation Accuracy\nModel: {self.__MODEL_NUMBER}")
        plt.plot(self.__HISTORY.history["accuracy"], "red", label="Accuracy")
        plt.plot(self.__HISTORY.history["val_accuracy"], "blue", label="Validation Accuracy")
        plt.legend()
        plt.savefig(f"{self.__DIRECTORY}/results/accuracy/model_{self.__MODEL_NUMBER}")
