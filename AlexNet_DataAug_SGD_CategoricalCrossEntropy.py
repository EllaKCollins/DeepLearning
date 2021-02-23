# AlexNet code 
# batch normalisation, dropout, with data augmentation optimizer is SGD, loss function is categorical crossentropy
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import seaborn as sns
import matplotlib.pyplot as mpl

start_img_size = 224
batch = 32
classes = 4 # number of classes

trainRoot = "archive/dataset2-master/dataset2-master/images/TRAIN"
testRoot = "archive/dataset2-master/dataset2-master/images/TEST"

model = tf.keras.Sequential([
	# 1st Convolutional Layer with ReLU activation
	layers.Conv2D(filters = 96, input_shape = (224, 224, 3), kernel_size = (11, 11), strides = (4, 4), padding = 'valid', activation = 'relu'),
	# Max-Pooling 
	layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'),
	# Batch Normalisation
	layers.BatchNormalization(),
	# 2nd Convolutional Layer with ReLU activation
	layers.Conv2D(filters = 256, kernel_size = (11, 11), strides = (1, 1), padding = 'valid', activation = 'relu'),
	# Max-Pooling
	layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'),
	# Batch Normalisation
	layers.BatchNormalization(),
	# 3rd Convolutional Layer with ReLU activation
	layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', activation = 'relu'),
	# Batch Normalisation
	layers.BatchNormalization(),
	# 4th Convolutional Layer with ReLU activation
	layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', activation = 'relu'),
	# Batch Normalisation
	layers.BatchNormalization(),
	# 5th Convolutional Layer with ReLU activation
	layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', activation = 'relu'),
	# Max-Pooling 
	layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'),
	# Batch Normalisation
	layers.BatchNormalization(),
	# Flattening 
	layers.Flatten(),
	# 1st Dense Layer 
	layers.Dense(4096, input_shape = (224*224*3, ), activation = 'relu'),
	# Add Dropout to prevent overfitting 
	layers.Dropout(0.4),
	# Batch Normalisation 
	layers.BatchNormalization(),
	# 2nd Dense Layer 
	layers.Dense(4096, activation = 'relu'),
	# Add Dropout 
	layers.Dropout(0.4),
	# Batch Normalisation 
	layers.BatchNormalization(),
	# Output Softmax Layer 
	layers.Dense(classes, activation = 'softmax')
])

testGen = ImageDataGenerator()
trainGen = ImageDataGenerator(validation_split=0.15)

testGenerator = testGen.flow_from_directory(testRoot,
                                            target_size=(224,224),
                                            shuffle=False
                                           )
trainGenerator = trainGen.flow_from_directory(trainRoot,
                                              subset="training",
                                              target_size=(224,224), 
                                             )
validationGenerator = trainGen.flow_from_directory(trainRoot,
                                                   subset="validation",
                                                   target_size=(224,224)
                                                  )

model.compile(optimizer = "SGD", loss= "categorical_crossentropy", metrics = ["accuracy"])

model.summary()

history = model.fit(trainGenerator,epochs=10,validation_data=validationGenerator)

y_pred = np.argmax(model.predict(testGenerator), axis=-1)

y_true = testGenerator.labels

print("Final test accuracy is {}%".format(accuracy_score(y_pred=y_pred,y_true=y_true)))

model.save('DataAug_SGD_CategoricalCrossEntropy')

confMatrix = confusion_matrix(y_pred=y_pred,y_true=y_true)

mpl.subplots(figsize=(6,6))
sns.heatmap(confMatrix,annot=True,fmt=".1f",linewidths=1.5)
mpl.xlabel("Predicted Label")
mpl.ylabel("Actual Label")
mpl.show()