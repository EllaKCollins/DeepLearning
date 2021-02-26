import numpy as np
import seaborn as sns
import matplotlib.pyplot as mpl
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


base_model = ResNet50(include_top=False,weights="imagenet")

for layer in base_model.layers[:140]:
    layer.trainable = False
    
model = keras.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Lambda(keras.applications.resnet50.preprocess_input),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4,activation="softmax")
])

testGen = ImageDataGenerator()
trainGen = ImageDataGenerator(validation_split=0.15
                             )

testingData = testGen.flow_from_directory("../DeepLearning/archive/dataset2-master/dataset2-master/images/TEST",
                                            target_size=(224,224),
                                            shuffle=False
                                           )
trainingData = trainGen.flow_from_directory("../DeepLearning/archive/dataset2-master/dataset2-master/images/TRAIN",
                                              subset="training",
                                              target_size=(224,224), 
                                             )
validationData = trainGen.flow_from_directory("../DeepLearning/archive/dataset2-master/dataset2-master/images/TRAIN",
                                                   subset="validation",
                                                   target_size=(224,224)
                                                  )


model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(trainingData,epochs=10,validation_data=validationData)

y_pred = model.predict_classes(testingData)

y_true = testingData.labels

print("Final test accuracy is {}%".format(accuracy_score(y_pred=y_pred,y_true=y_true)))

