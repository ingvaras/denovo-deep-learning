import json
from enum import Enum

import tensorflow as tf

from lib.building_blocks import classifier, conv_block, inception_module, residual_module, dense_module, \
    patch_extractor, patch_encoder, vision_transformer_module, multi_layer_perceptron


class DeNovoInception:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks, mutation_type):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.num_blocks = num_blocks
        self.num_modules = num_modules
        self.mutation_type = mutation_type

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        x = conv_block(inputs, self.filter_multiplier, (7, 7), strides=(2, 2))
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        aux_outputs = []

        for i in range(self.num_blocks):
            for j in range(self.num_modules):
                if j == 0 and i != 0:
                    aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
                    aux_conv = conv_block(aux_pool, 4 * self.filter_multiplier, (1, 1))
                    aux_outputs.append(classifier(aux_conv, name='auxiliary', j=i))
                    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
                x = inception_module(x, self.filter_multiplier * (2 ** i))

        final_output = classifier(x)

        return tf.keras.Model(inputs, outputs=[final_output, aux_outputs], name='DeNovoInception_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        filter_multiplier = trial.suggest_categorical('filter_multiplier', [2, 4, 8, 16, 32, 64])
        num_modules = trial.suggest_int('num_modules', 1, 6)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)

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

        x = conv_block(inputs, self.filter_multiplier, (7, 7), strides=(2, 2))
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        for i in range(self.num_blocks):
            for j in range(self.num_modules):
                x = residual_module(x, filter_multiplier=self.filter_multiplier * (2 ** i),
                                    increase_depth=i == 0,
                                    decrease_size=j == 0 and i != 0)

        final_output = classifier(x)

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoResNet_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        filter_multiplier = trial.suggest_categorical('filter_multiplier', [2, 4, 8, 16, 32, 64])
        num_modules = trial.suggest_int('num_modules', 1, 6)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)

        return DeNovoResNet(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                            num_blocks=num_blocks, mutation_type=mutation_type).model()

    @staticmethod
    def model_with_params_from_json(file_path, mutation_type):

        model = DeNovoResNet(input_shape=(164, 160, 3), filter_multiplier=None, num_modules=None,
                             num_blocks=None, mutation_type=mutation_type)

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

        x = conv_block(inputs, self.filter_multiplier, (7, 7), strides=(2, 2))
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        for i in range(self.num_blocks):
            for j in range(self.num_modules):
                if j == 0 and i != 0:
                    x = conv_block(x, self.filter_multiplier * (2 ** i) * 2, (1, 1),
                                   strides=(1, 1), reverse=True)
                    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)
                x = dense_module(x, filter_multiplier=self.filter_multiplier * (2 ** i))

        final_output = classifier(x)

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoDenseNet_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        filter_multiplier = trial.suggest_categorical('filter_multiplier', [2, 4, 8, 16, 32, 64])
        num_modules = trial.suggest_int('num_modules', 1, 6)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)

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

        final_output = tf.keras.layers.Dense(1, activation='sigmoid', name="main_classifier_0")(x)

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoViT_' + self.mutation_type)

    @staticmethod
    def model_with_suggested_parameters(trial, mutation_type):
        projection_dim = 2 ** trial.suggest_int('projection_dim_exp', 2, 6)
        num_heads = 2 ** trial.suggest_int('num_heads_exp', 0, 3)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)
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
