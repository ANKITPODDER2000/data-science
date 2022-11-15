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
