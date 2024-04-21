import gc
import json

import optuna
import tensorflow as tf

from lib.constants import MutationType, LEARNING_RATE
from lib.metrics import F1Score
from lib.models import Model

EPOCHS = 30
N_OF_TRIALS = 20

model_to_train = Model.DeNovoViT
for mutation_type in MutationType:
    train_data = tf.keras.preprocessing.image_dataset_from_directory('data/' + mutation_type.value + '/train',
                                                                     image_size=(164, 160))
    val_data = tf.keras.preprocessing.image_dataset_from_directory('data/' + mutation_type.value + '/val',
                                                                   image_size=(164, 160))


    def objective(trial):
        model = model_to_train.value.model_with_suggested_parameters(trial, mutation_type=mutation_type.value)

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(LEARNING_RATE),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[F1Score()]
        )

        model.fit(
            train_data,
            batch_size=32,
            validation_data=val_data,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)]
        )

        val_metrics = model.evaluate(val_data, verbose=0, return_dict=True)
        if 'main_classifier_0_f1_score' in val_metrics.keys():
            return val_metrics['main_classifier_0_f1_score']
        return val_metrics['f1_score']


    study = optuna.create_study(direction='maximize')
    print('Optimizing ' + model_to_train.value.__name__ + '_' + mutation_type.value)
    study.optimize(objective, n_trials=N_OF_TRIALS)
    with open('hyperparameters/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_best_params.json',
              'w') as f:
        best_params = study.best_params.copy()
        for param in list(best_params):
            if '_exp' in param:
                exp_param = param.replace('_exp', '')
                best_params[exp_param] = 2 ** best_params[param]
                del best_params[param]
        json.dump(best_params, f)

    gc.collect()
