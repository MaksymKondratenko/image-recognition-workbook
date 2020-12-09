import tensorflow as tf
import tensorflow.keras.layers as lay
import numpy as np
from moles.utils import load_data
import time

X_train, Y_train = load_data(0, 1000)
X_test, Y_test = load_data(1000, 1010)

model = tf.keras.models.Sequential([
    lay.Conv2D(8, (3,3), activation='relu', input_shape=(750, 1000, 3)),    # 765 x 1020
    lay.MaxPooling2D(2,2),  # 382 x 510
    lay.Conv2D(16, (3,3), activation='relu'),   # 380 x 508
    lay.MaxPooling2D(2,2),  # 190 x 254
    lay.Conv2D(32, (3,3), activation='relu'),   # 188 x 252
    lay.MaxPooling2D(2,2),  # 94 x 126
    lay.Conv2D(64, (3,3), activation='relu'),   # 92 x 124
    lay.MaxPooling2D(2,2),   # 46 x 62
    lay.Conv2D(64, (3,3), activation='relu'),   # 44 x 60
    lay.MaxPooling2D(2,2),  # 22 x 30
    lay.Conv2D(64, (3,3), activation='relu'),   # 20 x 28
    lay.MaxPooling2D(2,2),  # 10 x 14
    lay.Flatten(),
    lay.Dense(50, activation='relu'),
    lay.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()

model.fit(x=X_train, y=Y_train, batch_size=5, epochs=20)

predictions = model.predict(tf.convert_to_tensor(X_test), batch_size=5)

print('Predictions: ' + str(np.squeeze(predictions) > 0.5))
print('Real labels: ' + str(Y_test == 1))
print(time.time())
print(time.clock())



