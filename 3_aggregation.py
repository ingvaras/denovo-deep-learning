import pickle

import tensorflow as tf

from lib.metrics import F1Score
from lib.models import Model

TEST = True
EPOCHS = 10

model_to_evaluate = Model.DeNovoResNet

model_substitution = tf.keras.saving.load_model('models/' + model_to_evaluate.value.__name__ + '_substitution.keras')
model_insertion = tf.keras.saving.load_model('models/' + model_to_evaluate.value.__name__ + '_insertion.keras')
model_deletion = tf.keras.saving.load_model('models/' + model_to_evaluate.value.__name__ + '_deletion.keras')
model_substitution.trainable = False
model_insertion.trainable = False
model_deletion.trainable = False

input_layer = tf.keras.layers.Input(shape=(164, 160, 3))

output_substitution = model_substitution(input_layer)
output_insertion = model_insertion(input_layer)
output_deletion = model_deletion(input_layer)
if model_to_evaluate == Model.DeNovoInception:
    output_substitution = output_substitution[0]
    output_insertion = output_insertion[0]
    output_deletion = output_deletion[0]

concatenated_outputs = tf.keras.layers.Concatenate(axis=-1)([output_substitution, output_insertion, output_deletion])
combined_classifier = tf.keras.layers.Dense(1, activation='sigmoid')
output = combined_classifier(concatenated_outputs)

combined_model = tf.keras.models.Model(inputs=input_layer, outputs=output)

if not TEST:
    combined_model.compile(
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

    combined_model.fit(train_data,
                       batch_size=32,
                       epochs=EPOCHS,
                       verbose=1)
    with open('models/' + model_to_evaluate.value.__name__ + '_aggregated_head.pkl', 'wb') as f:
        pickle.dump(combined_model.layers[-1].get_weights(), f)
else:
    with open('models/' + model_to_evaluate.value.__name__ + '_aggregated_head.pkl', 'rb') as f:
        combined_classifier.set_weights(pickle.load(f))

    combined_model.compile(
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

    print(combined_model.evaluate(test_data, return_dict=True))
