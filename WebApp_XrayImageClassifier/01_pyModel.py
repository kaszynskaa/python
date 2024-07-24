import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np

# Define your local paths
root_path = './Data'
train_path = './Data/train'
test_path = './Data/test'

train_cov = os.path.join(train_path, 'COVID19')
train_normal = os.path.join(train_path, 'NORMAL')
test_cov = os.path.join(test_path, 'COVID19')
test_normal = os.path.join(test_path, 'NORMAL')

# Data preprocessing
train_generator = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2, zoom_range=0.1, rotation_range=12)
test_generator = ImageDataGenerator(rescale=1 / 255.0)

train_data = train_generator.flow_from_directory(train_path, target_size=(224, 224), subset='training', batch_size=64, class_mode='binary')
validation_data = train_generator.flow_from_directory(train_path, target_size=(224, 224), batch_size=64, subset='validation', class_mode='binary')
test_data = test_generator.flow_from_directory(test_path, target_size=(224, 224), batch_size=1, class_mode='binary', shuffle=False)

# Model architecture
resnet_base = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in resnet_base.layers:
    layer.trainable = False

resnet_out = resnet_base.get_layer('conv5_block1_2_bn').output
x = Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.21), padding='same')(resnet_out)
x = BatchNormalization(momentum=0.9)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(784, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = Dense(256, activation='relu')(x)
out_layer = Dense(1, activation='sigmoid')(x)  # or 2 with softmax

last_model = tf.keras.models.Model(inputs=resnet_base.input, outputs=out_layer)

last_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
lr_sch = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=np.sqrt(0.2), verbose=1, min_lr=5e-10)

# Model training
history = last_model.fit(train_data, epochs=24, validation_data=validation_data, batch_size=64, callbacks=[es, lr_sch])

# Save the model
last_model.save('model.h5')

# Save visualizations
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(acc, label='train acc')
plt.plot(val_acc, label='val_acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Accuracy_Graph.png')
plt.figure()

plt.plot(loss, label='TrainLoss')
plt.plot(val_loss, label='Val_loss')
plt.title('Training and validation loss')
plt.savefig('Loss_Graph.png')
plt.legend()

# Evaluation
true_labels = test_data.classes
predict = last_model.predict(test_data, steps=len(test_data.filenames))
y_pred = [1 * (x[0] >= 0.5) for x in predict]

# Classification report
report = classification_report(true_labels, y_pred, target_names=['COVID-19', 'NORMAL'], output_dict=True)
with open('classification_report.txt', 'w') as f:
    f.write(str(report))
