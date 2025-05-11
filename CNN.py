from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from Agent import Agent
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from Result import Result

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16
EPOCH = 200
DIRECTORY = "D:/Github/DermiumNet"
DATASET_DIRECTORY = DIRECTORY + "/datasets/isic2019/dataset/"  # "/datasets/flower/modelCrashDebugSet/"  # "/datasets/flower/flowersTestAugmentSplit/"
OUTPUT_UNIT = 8

trainDatagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
testDatagen = ImageDataGenerator(rescale=1.0 / 255)

trainingSet = trainDatagen.flow_from_directory(DATASET_DIRECTORY + "train", batch_size=BATCH_SIZE, class_mode="categorical")
validationSet = testDatagen.flow_from_directory(DATASET_DIRECTORY + "validation", batch_size=BATCH_SIZE, class_mode="categorical")
testSet = testDatagen.flow_from_directory(DATASET_DIRECTORY + "test", batch_size=10, class_mode="categorical", shuffle=False)

CLASS_LABELS = list(testSet.class_indices.keys())
TRUE_CLASSES = testSet.classes


class CNN:
    @staticmethod
    def getModel(agent: Agent) -> Sequential:
        model = Sequential()
        for i in range(agent.hyperparameters.convolution):
            if i == 0:
                model.add(Conv2D(filters=agent.hyperparameters.filters[i], kernel_size=(3, 3), activation="relu", padding="same", input_shape=INPUT_SHAPE))
            else:
                model.add(Conv2D(filters=agent.hyperparameters.filters[i], kernel_size=(3, 3), activation="relu", padding="same"))
            model.add(BatchNormalization())

            if i % 2 == 1:
                model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(GlobalAveragePooling2D())

        for i in range(agent.hyperparameters.dense):
            model.add(Dense(units=agent.hyperparameters.neurons[i], activation="relu"))
            model.add(Dropout(0.25))

        model.add(Dense(units=OUTPUT_UNIT, activation="softmax"))
        return model

    @staticmethod
    def getModelResult(agentNumber, model: Sequential) -> Result:
        model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        reduceLearningRate = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=1e-4, verbose=1)
        checkpoint = ModelCheckpoint(filepath=f"results/checkpoints/model_{agentNumber}.h5", monitor="val_loss", save_best_only=True, verbose=1)
        history = model.fit(trainingSet, steps_per_epoch=trainingSet.samples // BATCH_SIZE, epochs=EPOCH, validation_data=validationSet, validation_steps=validationSet.n // validationSet.batch_size, callbacks=[reduceLearningRate, checkpoint])
        predictions = model.predict(testSet, steps=testSet.samples // testSet.batch_size)
        result = Result(DIRECTORY, agentNumber, history, predictions, TRUE_CLASSES, CLASS_LABELS)
        clear_session()
        tf.keras.backend.clear_session()
        return result
