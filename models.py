import tensorflow as tf

from building_blocks import classifier, conv_block, inception_module, residual_module, dense_module


def denovo_cnn_v2(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = conv_block(inputs, 32, 3, strides=2, padding='valid')
    x = conv_block(x, 32, 3, padding='valid')
    x = conv_block(x, 64, 3)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, output, name='DeNovoCNN_v2')


class DeNovoInception:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks,
                 dropout_rate):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_modules = num_modules

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        x = conv_block(inputs, self.filter_multiplier, (7, 7), strides=(2, 2))
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        aux_outputs = []

        for i in range(self.num_blocks):
            for j in range(self.num_modules):
                x = inception_module(x, self.filter_multiplier)
                if j == 0 and i != 0:
                    aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
                    aux_conv = conv_block(aux_pool, 4 * self.filter_multiplier, (1, 1))
                    aux_outputs.append(classifier(aux_conv, name='auxiliary', j=i))

        final_output = classifier(x, dropout_rate=self.dropout_rate)

        return tf.keras.Model(inputs, outputs=[final_output, aux_outputs], name='DeNovoInception')

    @staticmethod
    def model_with_suggested_parameters(trial):
        filter_multiplier = 2 ** trial.suggest_int('filter_multiplier_exp', 1, 6)
        num_modules = trial.suggest_int('num_modules', 1, 5)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)
        dropout_rate = trial.suggest_int('dropout_rate', 0, 500) / 1000

        return DeNovoInception(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                               num_blocks=num_blocks, dropout_rate=dropout_rate).model()


class DeNovoResNet:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.num_modules = num_modules
        self.num_blocks = num_blocks

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

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoResNet')

    @staticmethod
    def model_with_suggested_parameters(trial):
        filter_multiplier = 2 ** trial.suggest_int('filter_multiplier_exp', 1, 6)
        num_modules = trial.suggest_int('num_modules', 1, 5)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)

        return DeNovoResNet(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                            num_blocks=num_blocks).model()


class DeNovoDenseNet:
    def __init__(self, input_shape, filter_multiplier, num_modules, num_blocks):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.num_modules = num_modules
        self.num_blocks = num_blocks

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

        return tf.keras.Model(inputs, outputs=[final_output], name='DeNovoDenseNet')

    @staticmethod
    def model_with_suggested_parameters(trial):
        filter_multiplier = 2 ** trial.suggest_int('filter_multiplier_exp', 1, 6)
        num_modules = trial.suggest_int('num_modules', 1, 5)
        num_blocks = trial.suggest_int('num_blocks', 1, 3)

        return DeNovoDenseNet(input_shape=(164, 160, 3), filter_multiplier=filter_multiplier, num_modules=num_modules,
                              num_blocks=num_blocks).model()


class Models:
    DeNovoInception = DeNovoInception
    DeNovoResNet = DeNovoResNet
    DeNovoDenseNet = DeNovoDenseNet
