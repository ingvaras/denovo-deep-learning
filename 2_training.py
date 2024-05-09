import math

import matplotlib.pyplot as plt
import tensorflow as tf

from lib.augmentation import CustomDataGenerator
from lib.constants import MutationType, LEARNING_RATE
from lib.metrics import F1Score
from lib.models import Model
from lib.utils import get_steps_per_epoch

EPOCHS = 200
model_to_train = Model.DeNovoInception

for mutation_type in MutationType:
    def step_decay(epoch):
        initial_learning_rate = LEARNING_RATE
        drop = 0.5
        epochs_drop = 10.0 if mutation_type != MutationType.Substitution else 6.0
        return initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))


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

    model = model_to_train.value.model_with_params_from_json(
        'hyperparameters/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_best_params.json',
        mutation_type=mutation_type.value)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[F1Score(), tf.keras.metrics.BinaryAccuracy()]
    )

    history = model.fit(
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

    plt.plot(
        history.history['f1_score' if model_to_train != Model.DeNovoInception else 'main_classifier_0_f1_score'])
    plt.plot(history.history[
                 'val_f1_score' if model_to_train != Model.DeNovoInception else 'val_main_classifier_0_f1_score'])
    plt.title(model_to_train.value.__name__ + ' ' + mutation_type.value + ' f1 score')
    plt.ylabel('F1 score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plots/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_f1_score.png')
    plt.show()

    plt.plot(history.history['loss' if model_to_train != Model.DeNovoInception else 'main_classifier_0_loss'])
    plt.plot(
        history.history['val_loss' if model_to_train != Model.DeNovoInception else 'val_main_classifier_0_loss'])
    plt.title(model_to_train.value.__name__ + ' ' + mutation_type.value + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('plots/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_loss.png')
    plt.show()

    model.save('models/' + model_to_train.value.__name__ + '_' + mutation_type.value + '.keras')
