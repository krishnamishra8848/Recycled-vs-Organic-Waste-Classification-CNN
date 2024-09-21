from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d techsash/waste-classification-data

!unzip waste-classification-data.zip -d waste_data

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import HeNormal

# Directories for the dataset
train_dir = '/content/waste_data/DATASET/TRAIN'

# Set validation split and image data generator
validation_split = 0.2
datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

# Load training data (80%)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Use binary since output is 1 (Sigmoid)
    subset='training'
)

# Load validation data (20%)
validation_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Build the CNN model
model = models.Sequential()

# First Conv Block (32 filters)
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal(), input_shape=(224, 224, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Second Conv Block (64 filters)
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal()))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Third Conv Block (128 filters)
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=HeNormal()))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Flatten layer
model.add(layers.Flatten())

# Dense layer with 128 units
model.add(layers.Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# Output layer (Sigmoid activation for binary classification)
model.add(layers.Dense(1, activation='sigmoid'))

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = callbacks.LearningRateScheduler(lr_scheduler)

# Early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Compile the model
model.compile(
    optimizer=optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    epochs=5,  # You can adjust the number of epochs
    batch_size=32,
    validation_data=validation_data,
    callbacks=[early_stopping, lr_callback]
)

# Evaluate the model
val_loss, val_acc = model.evaluate(validation_data)
print(f"Validation accuracy: {val_acc}")

# Save the model if needed
model.save('waste_classification_cnn.h5')

Found 18052 images belonging to 2 classes.
Found 4512 images belonging to 2 classes.
/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/5
/usr/local/lib/python3.10/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
565/565 ━━━━━━━━━━━━━━━━━━━━ 64s 90ms/step - accuracy: 0.7816 - loss: 0.5197 - val_accuracy: 0.8287 - val_loss: 0.3872 - learning_rate: 0.0010
Epoch 2/5
565/565 ━━━━━━━━━━━━━━━━━━━━ 69s 83ms/step - accuracy: 0.8390 - loss: 0.3898 - val_accuracy: 0.8203 - val_loss: 0.4051 - learning_rate: 0.0010
Epoch 3/5
565/565 ━━━━━━━━━━━━━━━━━━━━ 43s 76ms/step - accuracy: 0.8588 - loss: 0.3354 - val_accuracy: 0.8336 - val_loss: 0.3844 - learning_rate: 0.0010
Epoch 4/5
565/565 ━━━━━━━━━━━━━━━━━━━━ 83s 78ms/step - accuracy: 0.8733 - loss: 0.3146 - val_accuracy: 0.7886 - val_loss: 0.4968 - learning_rate: 0.0010
Epoch 5/5
565/565 ━━━━━━━━━━━━━━━━━━━━ 80s 76ms/step - accuracy: 0.8852 - loss: 0.2852 - val_accuracy: 0.8712 - val_loss: 0.3224 - learning_rate: 0.0010
141/141 ━━━━━━━━━━━━━━━━━━━━ 7s 52ms/step - accuracy: 0.8702 - loss: 0.3177
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Validation accuracy: 0.871232271194458

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('/content/waste_classification_cnn.h5')

# Preprocess the test image
image_path = '/content/waste_data/DATASET/TEST/R/R_10006.jpg'

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize the image
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Preprocess and predict the image class
test_image = preprocess_image(image_path)
prediction = model.predict(test_image)

# Determine class
if prediction < 0.5:
    print("The image is classified as Organic waste (O).")
else:
    print("The image is classified as Recyclable waste (R).")
