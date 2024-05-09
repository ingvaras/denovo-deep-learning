import time

import numpy as np
import tensorflow as tf

from lib.constants import MutationType
from lib.metrics import F1Score
from lib.models import Model

model_to_evaluate = Model.DeNovoInception

for mutation_type in MutationType:
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_std_normalization=True,
                                                                   samplewise_center=True)

    test_generator = test_datagen.flow_from_directory(
        'data/' + mutation_type.value + '/test',
        target_size=(164, 160),
        batch_size=32,
        classes=['IV', 'DNM'],
        class_mode='binary')

    model = tf.keras.saving.load_model(
        'models/' + model_to_evaluate.value.__name__ + '_' + mutation_type.value + '.keras',
        custom_objects={"F1Score": F1Score})

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            F1Score()]
    )
    print(mutation_type.value)
    model.evaluate(test_generator)

    input_data = np.random.randn(1280, 164, 160, 3).astype(np.float32)
    # warmup
    model.predict(input_data, verbose=0)

    start_time = time.time()
    model.predict(input_data, verbose=0)
    end_time = time.time()

    print("Inference time:", (end_time - start_time) / 1280 * 1000, "ms")
