import json
from enum import Enum

import numpy as np
import tensorflow as tf

from lib.building_blocks import classifier, conv_block, inception_module, patch_extractor, \
    patch_encoder, vision_transformer_module, multi_layer_perceptron, squeeze_excite_block, dense_module
from lib.constants import L1, POSITIVE_NEGATIVE_RATIO


class DeNovoInception:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks, mutation_type):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.num_blocks = num_blocks
        self.num_modules = num_modules
        self.mutation_type = mutation_type

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        aux_outputs = []
        x = inputs

        for i in range(self.num_blocks):
            for j in range(self.num_modules):
                x = inception_module(x, self.filter_multiplier + 32 * i)
                x = squeeze_excite_block(x, x.shape[3])
            if i != self.num_blocks - 1:
                aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
                aux_conv = conv_block(aux_pool, self.filter_multiplier // 2, (1, 1))
                aux_outputs.append(classifier(aux_conv, name='auxiliary', j=i))
                x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        final_output = classifier(x)

        return tf.keras.Model(inputs, outputs=[final_output, aux_outputs], name='DeNovoInception_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        filter_multiplier = trial.suggest_categorical('filter_multiplier', [16, 32])
        num_modules = trial.suggest_int('num_modules', 2, 3)
        num_blocks = trial.suggest_int('num_blocks', 2, 4)

        return DeNovoInception(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                               num_blocks=num_blocks, mutation_type=mutation_type).model()

    @staticmethod
    def model_with_params_from_json(file_path, mutation_type):

        model = DeNovoInception(input_shape=(164, 160, 3), filter_multiplier=None, num_modules=None,
                                num_blocks=None, mutation_type=mutation_type)

        with open(file_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(model, key, value)
        return model.model()


class DeNovoResNet:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks, mutation_type):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.num_modules = num_modules
        self.num_blocks = num_blocks
        self.mutation_type = mutation_type

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        x = inputs
        for i in range(self.num_blocks):
            shortcut = conv_block(x, self.filter_multiplier + 32 * i, (1, 1), strides=(2, 2), activation='linear')
            for j in range(self.num_modules):
                x = conv_block(x, self.filter_multiplier + 32 * i, (3, 3), activation='relu')
            x = squeeze_excite_block(x, self.filter_multiplier + 32 * i)
            x = tf.keras.layers.AveragePooling2D((2, 2))(x)
            if shortcut.shape != x.shape:
                x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 0)))(x)
            x = tf.keras.layers.add([shortcut, x])

        final_output = classifier(x)

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoResNet_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        filter_multiplier = trial.suggest_categorical('filter_multiplier', [32, 64, 96])
        num_modules = trial.suggest_int('num_modules', 1, 4)
        num_blocks = trial.suggest_int('num_blocks', 1, 4)

        return DeNovoResNet(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                            num_blocks=num_blocks, mutation_type=mutation_type).model()

    @staticmethod
    def model_with_params_from_json(file_path, mutation_type):
        model = DeNovoResNet(input_shape=(164, 160, 3), filter_multiplier=None, num_modules=None, num_blocks=None,
                             mutation_type=mutation_type)

        with open(file_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(model, key, value)
        return model.model()


class DeNovoDenseNet:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks, mutation_type):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.num_modules = num_modules
        self.num_blocks = num_blocks
        self.mutation_type = mutation_type

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        x = inputs

        for i in range(self.num_blocks):
            for j in range(self.num_modules):
                x = dense_module(x, filter_multiplier=self.filter_multiplier + 32 * i)
            x = conv_block(x, self.filter_multiplier + 32 * (i + 1), (1, 1),
                           strides=(1, 1), reverse=True)
            x = squeeze_excite_block(x, self.filter_multiplier + 32 * (i + 1))
            x = tf.keras.layers.AveragePooling2D((2, 2))(x)

        final_output = classifier(x)

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoDenseNet_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        filter_multiplier = trial.suggest_categorical('filter_multiplier', [32, 64, 96])
        num_modules = trial.suggest_int('num_modules', 1, 4)
        num_blocks = trial.suggest_int('num_blocks', 1, 4)

        return DeNovoDenseNet(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                              num_blocks=num_blocks, mutation_type=mutation_type).model()

    @staticmethod
    def model_with_params_from_json(file_path, mutation_type):

        model = DeNovoDenseNet(input_shape=(164, 160, 3), filter_multiplier=None, num_modules=None,
                               num_blocks=None, mutation_type=mutation_type)

        with open(file_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(model, key, value)
        return model.model()


class DeNovoViT:
    def __init__(self, input_shape, projection_dim, num_heads, num_blocks, patch_size,
                 mutation_type):
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.mutation_type = mutation_type

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        patches = patch_extractor(inputs, self.patch_size)
        x = patch_encoder(patches, self.projection_dim)

        for i in range(self.num_blocks):
            x = vision_transformer_module(x, self.num_heads, self.projection_dim)

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Flatten()(x)
        x = multi_layer_perceptron(x, self.projection_dim)

        final_output = tf.keras.layers.Dense(1, activation='sigmoid',
                                             kernel_initializer=tf.keras.initializers.HeNormal(),
                                             name="main_classifier_0",
                                             bias_initializer=tf.keras.initializers.Constant(
                                                 np.log([POSITIVE_NEGATIVE_RATIO])),
                                             kernel_regularizer=tf.keras.regularizers.l1(L1))(x)

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoViT_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        projection_dim = trial.suggest_categorical('projection_dim', [32, 64, 96, 128])
        num_heads = 2 ** trial.suggest_int('num_heads_exp', 0, 3)
        num_blocks = trial.suggest_int('num_blocks', 1, 10)
        patch_size = trial.suggest_categorical('patch_size', [16, 20])

        return DeNovoViT(input_shape=(164, 160, 3), projection_dim=projection_dim, num_heads=num_heads,
                         num_blocks=num_blocks, patch_size=patch_size, mutation_type=mutation_type).model()

    @staticmethod
    def model_with_params_from_json(file_path, mutation_type):

        model = DeNovoViT(input_shape=(164, 160, 3), projection_dim=None, num_heads=None, num_blocks=None,
                          patch_size=None, mutation_type=mutation_type)

        with open(file_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(model, key, value)
        return model.model()


class Model(Enum):
    DeNovoInception = DeNovoInception
    DeNovoResNet = DeNovoResNet
    DeNovoDenseNet = DeNovoDenseNet
    DeNovoViT = DeNovoViT
