import pickle

import tensorflow as tf

from lib.constants import MutationType
from lib.metrics import F1Score
from lib.models import Model

TEST = True
EPOCHS = 10

input_layer = tf.keras.layers.Input(shape=(164, 160, 3))
model_outputs = []

for model in Model:
    for mutation_type in MutationType:
        loaded_model = tf.keras.saving.load_model(
            'models/' + model.value.__name__ + '_' + mutation_type.value + '.keras')
        loaded_model.trainable = False
        model_outputs.append(
            loaded_model(input_layer) if model != Model.DeNovoInception else loaded_model(input_layer)[0])

concatenated_outputs = tf.keras.layers.Concatenate(axis=-1)(model_outputs)
combined_classifier = tf.keras.layers.Dense(1, activation='sigmoid')
output = combined_classifier(concatenated_outputs)

ensemble_model = tf.keras.models.Model(inputs=input_layer, outputs=output)

if not TEST:
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[F1Score()]
    )

    train_data_substitutions = tf.keras.preprocessing.image_dataset_from_directory('data/substitution/train',
                                                                                   image_size=(164, 160))
    val_data_substitutions = tf.keras.preprocessing.image_dataset_from_directory('data/substitution/val',
                                                                                 image_size=(164, 160))
    train_data_insertions = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/train',
                                                                                image_size=(164, 160))
    val_data_insertions = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/val',
                                                                              image_size=(164, 160))
    train_data_deletions = tf.keras.preprocessing.image_dataset_from_directory('data/deletion/train',
                                                                               image_size=(164, 160))
    val_data_deletions = tf.keras.preprocessing.image_dataset_from_directory('data/deletion/val', image_size=(164, 160))

    train_data = train_data_substitutions.concatenate(train_data_insertions).concatenate(
        train_data_deletions).concatenate(val_data_substitutions).concatenate(val_data_insertions).concatenate(
        val_data_deletions)

    train_data = train_data.shuffle(len(train_data))

    ensemble_model.fit(train_data,
                       batch_size=32,
                       epochs=EPOCHS,
                       verbose=1)
    with open('models/DeNovoEnsemble_aggregated_head.pkl', 'wb') as f:
        pickle.dump(ensemble_model.layers[-1].get_weights(), f)
else:
    with open('models/DeNovoEnsemble_aggregated_head.pkl', 'rb') as f:
        combined_classifier.set_weights(pickle.load(f))

    ensemble_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            F1Score()]
    )

    test_data_substitutions = tf.keras.preprocessing.image_dataset_from_directory('data/substitution/test',
                                                                                  image_size=(164, 160))
    test_data_insertions = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/test',
                                                                               image_size=(164, 160))
    test_data_deletions = tf.keras.preprocessing.image_dataset_from_directory('data/deletion/test',
                                                                              image_size=(164, 160))

    test_data = test_data_substitutions.concatenate(test_data_insertions).concatenate(test_data_deletions)

    print(ensemble_model.evaluate(test_data, return_dict=True))
