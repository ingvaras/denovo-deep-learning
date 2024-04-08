import tensorflow as tf

from models import DeNovoInceptionV4

TEST = False
EPOCHS = 25
LEARNING_RATE = 3e-4

train_data = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/train', image_size=(164, 160))
val_data = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/val', image_size=(164, 160))
test_data = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/test', image_size=(164, 160))

model = DeNovoInceptionV4(input_shape=(164, 160, 3), filter_multiplier=1, num_a_modules=0, num_b_modules=0,
                          num_c_modules=0, auxiliary_outputs=True, dropout_rate=0).model()

tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")]),
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS)
