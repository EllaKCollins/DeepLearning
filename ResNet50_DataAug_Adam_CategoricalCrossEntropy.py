import numpy as np
import seaborn as sns
import matplotlib.pyplot as mpl
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score,confusion_matrix


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

#model.add(layers.Input(shape=(224,224,3)))
#model.add(layers.Lambda(keras.applications.resnet50.preprocess_input))
#model.add(base_model)
#model.add(layers.GlobalAveragePooling2D())
#model.add()

testGen = ImageDataGenerator()
trainGen = ImageDataGenerator(validation_split=0.15,
                              vertical_flip=True,
                              horizontal_flip=True,
                              rotation_range=0.15
                             )

testGenerator = testGen.flow_from_directory("/Users/manonheinhuis/Desktop/Uni/Year4AI/2A/Deep Learning/DeepLearning/archive/dataset2-master/dataset2-master/images/TEST",
                                            target_size=(224,224),
                                            shuffle=False
                                           )
trainGenerator = trainGen.flow_from_directory("/Users/manonheinhuis/Desktop/Uni/Year4AI/2A/Deep Learning/DeepLearning/archive/dataset2-master/dataset2-master/images/TRAIN",
                                              subset="training",
                                              target_size=(224,224), 
                                             )
validationGenerator = trainGen.flow_from_directory("/Users/manonheinhuis/Desktop/Uni/Year4AI/2A/Deep Learning/DeepLearning/archive/dataset2-master/dataset2-master/images/TRAIN",
                                                   subset="validation",
                                                   target_size=(224,224)
                                                  )


model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(trainGenerator,epochs=1,validation_data=validationGenerator)

y_pred = model.predict_classes(testGenerator)

y_true = testGenerator.labels

print("Final test accuracy is {}%".format(accuracy_score(y_pred=y_pred,y_true=y_true)))

confMatrix = confusion_matrix(y_pred=y_pred,y_true=y_true)

mpl.subplots(figsize=(6,6))
sns.heatmap(confMatrix,annot=True,fmt=".1f",linewidths=1.5)
mpl.xlabel("Predicted Label")
mpl.ylabel("Actual Label")
mpl.show()
