import json

import optuna
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

from models import Models

TEST = False
EPOCHS = 20
N_OF_TRIALS = 20
model_to_train = Models.DeNovoDenseNet

train_data_substitutions = tf.keras.preprocessing.image_dataset_from_directory('data/substitution/train',
                                                                               image_size=(164, 160))
val_data_substitutions = tf.keras.preprocessing.image_dataset_from_directory('data/substitution/val',
                                                                             image_size=(164, 160))
test_data_substitutions = tf.keras.preprocessing.image_dataset_from_directory('data/substitution/test',
                                                                              image_size=(164, 160))
train_data_insertions = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/train',
                                                                            image_size=(164, 160))
val_data_insertions = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/val', image_size=(164, 160))
test_data_insertions = tf.keras.preprocessing.image_dataset_from_directory('data/insertion/test', image_size=(164, 160))
train_data_deletions = tf.keras.preprocessing.image_dataset_from_directory('data/deletion/train', image_size=(164, 160))
val_data_deletions = tf.keras.preprocessing.image_dataset_from_directory('data/deletion/val', image_size=(164, 160))
test_data_deletions = tf.keras.preprocessing.image_dataset_from_directory('data/deletion/test', image_size=(164, 160))

model = tf.keras.applications.InceptionV3()
tf.keras.utils.plot_model(model, show_shapes=True)


def objective(trial):
    model = model_to_train.model_with_suggested_parameters(trial)

    print(model.summary())
    # tf.keras.utils.plot_model(model, show_shapes=True)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="f1_score")]
    )

    f1_score_average = 0
    for data in [(train_data_substitutions, val_data_substitutions), (train_data_insertions, val_data_insertions),
                 (train_data_deletions, val_data_deletions)]:
        model.fit(
            data[0],
            batch_size=32,
            validation_data=data[1],
            epochs=EPOCHS,
            verbose=1,
            callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)]
        )

        val_metrics = model.evaluate(data[1], verbose=0, return_dict=True)
        if 'main_classifier_0_f1_score' in val_metrics.keys():
            f1_score_average += val_metrics['main_classifier_0_f1_score']
        else:
            f1_score_average += val_metrics['f1_score']
    return f1_score_average / 3


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_OF_TRIALS)
with open(model_to_train.__name__ + '_best_params.json', 'w') as f:
    json.dump(study.best_params, f)

best_params = study.best_params
print("Best params:", best_params)
