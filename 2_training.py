import matplotlib.pyplot as plt
import tensorflow as tf

from lib.constants import MutationType, LEARNING_RATE
from lib.metrics import F1Score
from lib.models import Model

EPOCHS = 200
model_to_train = Model.DeNovoInception

for mutation_type in MutationType:
    train_data = tf.keras.preprocessing.image_dataset_from_directory('data/' + mutation_type.value + '/train',
                                                                     image_size=(164, 160))
    val_data = tf.keras.preprocessing.image_dataset_from_directory('data/' + mutation_type.value + '/val',
                                                                   image_size=(164, 160))

    model = model_to_train.value.model_with_params_from_json(
        'hyperparameters/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_best_params.json',
        mutation_type=mutation_type.value)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[F1Score()]
    )

    # print(model.summary())
    # tf.keras.utils.plot_model(model, show_shapes=True)

    history = model.fit(
        train_data,
        batch_size=32,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if model_to_train != Model.DeNovoInception else 'val_main_classifier_0_loss',
            patience=5, verbose=1, restore_best_weights=True)]
    )

    plt.plot(history.history['f1_score' if model_to_train != Model.DeNovoInception else 'main_classifier_0_f1_score'])
    plt.plot(history.history[
                 'val_f1_score' if model_to_train != Model.DeNovoInception else 'val_main_classifier_0_f1_score'])
    plt.title(model_to_train.value.__name__ + ' ' + mutation_type.value + ' f1 score')
    plt.ylabel('F1 score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plots/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_f1_score.png')
    plt.show()

    plt.plot(history.history['loss' if model_to_train != Model.DeNovoInception else 'main_classifier_0_loss'])
    plt.plot(history.history['val_loss' if model_to_train != Model.DeNovoInception else 'val_main_classifier_0_loss'])
    plt.title(model_to_train.value.__name__ + ' ' + mutation_type.value + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('plots/' + model_to_train.value.__name__ + '_' + mutation_type.value + '_loss.png')
    plt.show()

    model.save('models/' + model_to_train.value.__name__ + '_' + mutation_type.value + '.keras')
