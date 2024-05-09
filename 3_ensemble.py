import pickle

import numpy as np
import tensorflow as tf

from lib.augmentation import CustomDataGenerator
from lib.constants import MutationType, POSITIVE_NEGATIVE_RATIO
from lib.metrics import F1Score
from lib.models import Model
from lib.utils import get_steps_per_epoch

TEST = False
EPOCHS = 20

input_layer = tf.keras.layers.Input(shape=(164, 160, 3))

for mutation_type in [MutationType.Insertion]:
    model_outputs = []
    for model in Model:
        loaded_model = tf.keras.saving.load_model(
            'models/' + model.value.__name__ + '_' + mutation_type.value + '.keras',
            custom_objects={"F1Score": F1Score})
        loaded_model.trainable = False
        model_outputs.append(
            loaded_model(input_layer) if model != Model.DeNovoInception else loaded_model(input_layer)[0])

    concatenated_outputs = tf.keras.layers.Concatenate(axis=-1)(model_outputs)
    combined_classifier = tf.keras.layers.Dense(1, activation='sigmoid',
                                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                                bias_initializer=tf.keras.initializers.Constant(
                                                    np.log([POSITIVE_NEGATIVE_RATIO])))
    output = combined_classifier(concatenated_outputs)

    ensemble_model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    if not TEST:
        # with open('models/DeNovoEnsemble_' + mutation_type.value + '_head.pkl', 'rb') as f:
        #    combined_classifier.set_weights(pickle.load(f))

        ensemble_model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[F1Score(), tf.keras.metrics.BinaryAccuracy()]
        )

        train_datagen = CustomDataGenerator(samplewise_std_normalization=True, samplewise_center=True,
                                            brightness_range=[0.3, 1.], horizontal_flip=True)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_std_normalization=True,
                                                                      samplewise_center=True)

        train_generator = train_datagen.flow_from_directory(
            'data/' + mutation_type.value + '/train',
            target_size=(164, 160),
            batch_size=32,
            classes=['IV', 'DNM'],
            class_mode='binary')

        validation_generator = val_datagen.flow_from_directory(
            'data/' + mutation_type.value + '/val',
            target_size=(164, 160),
            batch_size=32,
            classes=['IV', 'DNM'],
            class_mode='binary')

        ensemble_model.fit(train_generator,
                           batch_size=32,
                           epochs=EPOCHS,
                           steps_per_epoch=get_steps_per_epoch('data/' + mutation_type.value + '/train'),
                           validation_data=validation_generator,
                           validation_steps=get_steps_per_epoch('data/' + mutation_type.value + '/val'),
                           verbose=1)
        with open('models/DeNovoEnsemble_' + mutation_type.value + '_head.pkl', 'wb') as f:
            pickle.dump(ensemble_model.layers[-1].get_weights(), f)
    else:
        with open('models/DeNovoEnsemble_' + mutation_type.value + '_head.pkl', 'rb') as f:
            combined_classifier.set_weights(pickle.load(f))

        ensemble_model.compile(
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

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_std_normalization=True,
                                                                       samplewise_center=True)

        test_generator = test_datagen.flow_from_directory(
            'data/' + mutation_type.value + '/test',
            target_size=(164, 160),
            batch_size=32,
            classes=['IV', 'DNM'],
            class_mode='binary')

        ensemble_model.evaluate(test_generator, return_dict=True)
