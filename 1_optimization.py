import gc
import json
import math

import optuna
import tensorflow as tf

from lib.augmentation import CustomDataGenerator
from lib.constants import MutationType, LEARNING_RATE
from lib.metrics import F1Score
from lib.models import Model
from lib.utils import get_steps_per_epoch

EPOCHS = 150
N_OF_TRIALS = 20

model_to_train = Model.DeNovoViT


def step_decay(epoch):
    initial_learning_rate = LEARNING_RATE
    drop = 0.5
    epochs_drop = 10.0 if mutation_type != MutationType.Substitution else 6.0
    return initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))


for mutation_type in [MutationType.Insertion]:
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


    def objective(trial):
        model = model_to_train.value.model_with_suggested_parameters(trial, mutation_type=mutation_type.value)

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[F1Score(), tf.keras.metrics.BinaryAccuracy()]
        )

        model.fit(
            train_generator,
            batch_size=32,
            steps_per_epoch=get_steps_per_epoch('data/' + mutation_type.value + '/train'),
            validation_data=validation_generator,
            validation_steps=get_steps_per_epoch('data/' + mutation_type.value + '/val'),
            epochs=EPOCHS,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=30 if mutation_type != MutationType.Substitution else 15,
                                                 verbose=1, restore_best_weights=True),
                tf.keras.callbacks.LearningRateScheduler(step_decay)]
        )

        val_metrics = model.evaluate(validation_generator, verbose=0, return_dict=True)
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
