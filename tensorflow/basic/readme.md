# MNIST Dataset
----

### Important Code

```
class CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs = {}):
        if logs['accuracy'] > 0.95:
            print(f"Model accuracy is more than {(logs['accuracy'] * 100)} %%")
            self.model.stop_training = True

cb = CallBack()
model4 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(1024, activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(10,  activation = tf.keras.activations.softmax)
])

model4.compile(
  optimizer = tf.keras.optimizers.Adam() , 
  loss = tf.keras.losses.sparse_categorical_crossentropy,
  metrics = ['accuracy']
)
model4.fit(training_data, training_label, epochs = 100, callbacks=[cb])
```

### Training Data
![traing_data_image](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/mnist.png)

### Loss | Accuracy Curve
![alt text](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/loss.png?raw=true)

### Prediction
![Prediction](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/prediction.png)


# MNIST - Conv2D | MaxPooling2D
----

### Important Code

```
class CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs = {}):
        if logs['accuracy'] > 0.95:
            print(f"Model accuracy is more than {(logs['accuracy'] * 100)} %%")
            self.model.stop_training = True

cb = CallBack()
model4 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), input_shape = (28,28,1), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(10,  activation = tf.keras.activations.softmax)
])

model4.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metrics = ['accuracy']
)

model4.fit(training_data, training_label, epochs = 100, callbacks=[cb])
```

### Loss | Accuracy Curve
![alt text](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/loss2.png)

### Prediction
![Prediction](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/prediction2.png)

# Horse Human Dataset
---
### Important Code - New Block of Codes

```
import zipfile
train_file = zipfile.ZipFile('/content/horse-or-human.zip', 'r')
test_file  = zipfile.ZipFile('/content/validation-horse-or-human.zip', 'r')

train_file.extractall('train-dir')
test_file.extractall('test-dir')

#----------------------------------------------------------------------#

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0)
train_data = train_data_gen.flow_from_directory(
    '/content/train-dir',
    target_size = (150, 150),
    batch_size = 128,
    class_mode = 'binary'
)
#----------------------------------------------------------------------#
history = model4.fit(
    train_data,
    steps_per_epoch = train_data.n // train_data.batch_size,
    epochs = 100,
    verbose = 1,
    validation_data = test_data,
    validation_steps = test_data.n // test_data.batch_size,
    callbacks = [cb]
)
```
### Loss | Accuracy Curve
![alt text](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/animation.gif)

### Prediction
![Prediction](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/horse_human.png)

# Horse Human Dataset - Augmentation
### Important Code - New Block of Codes
```
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1.0/255.0,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    zoom_range = 0.3,
    shear_range = 0.3,
    horizontal_flip = True,
    fill_mode = 'nearest'
    
)
```
### Augmentation
![alt text](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/Augmented_image.png)

### Loss | Accuracy Curve
![alt text](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/animation-aug-horse-human.gif)

### Prediction
![Prediction](https://github.com/ANKITPODDER2000/data-science/blob/main/tensorflow/basic/image/pred_aug.png)
