import tensorflow as tf

from building_blocks import conv_block, inception_module_a, inception_module_b, inception_module_c, \
    inception_reduction_module_a, inception_reduction_module_b


def denovo_cnn_v2(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = conv_block(inputs, 32, 3, strides=2, padding='valid')
    x = conv_block(x, 32, 3, padding='valid')
    x = conv_block(x, 64, 3)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, output, name='DeNovoCNN_v2')


class DeNovoInceptionV4:
    def __init__(self, input_shape, filter_multiplier, num_a_modules, num_b_modules, num_c_modules, pooling='avg',
                 dropout_rate=False, auxiliary_outputs=False):
        self.input_shape = input_shape
        self.filter_multiplier = filter_multiplier
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxiliary_outputs = auxiliary_outputs
        self.num_a_modules = num_a_modules
        self.num_b_modules = num_b_modules
        self.num_c_modules = num_c_modules

    def classifier(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return outputs

    def model(self):
        inputs = tf.keras.Input(self.input_shape)

        x = conv_block(inputs, self.filter_multiplier, 3, strides=2, padding='valid')
        x = conv_block(x, self.filter_multiplier, 3, padding='valid')
        x = conv_block(x, 2 * self.filter_multiplier, 3)

        branch1 = conv_block(x, 3 * self.filter_multiplier, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = conv_block(x, 2 * self.filter_multiplier, 1)
        branch1 = conv_block(branch1, 3 * self.filter_multiplier, 3, padding='valid')
        branch2 = conv_block(x, 2 * self.filter_multiplier, 1)
        branch2 = conv_block(branch2, 2 * self.filter_multiplier, 7)
        branch2 = conv_block(branch2, 3 * self.filter_multiplier, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = conv_block(x, 6 * self.filter_multiplier, 3, strides=(2, 2), padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=1)

        for i in range(self.num_a_modules):
            x = inception_module_a(x, self.filter_multiplier, i)

        aux_output_0 = []
        if self.auxiliary_outputs:
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = conv_block(aux_pool, 3 * self.filter_multiplier, 1)
            aux_output_0 = self.classifier(aux_conv)

        x = inception_reduction_module_a(x, self.filter_multiplier, 1)

        for i in range(self.num_b_modules):
            x = inception_module_b(x, self.filter_multiplier, i)

        aux_output_1 = []
        if self.auxiliary_outputs:
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = conv_block(aux_pool, 4 * self.filter_multiplier, 1)
            aux_output_1 = self.classifier(aux_conv)

        x = inception_reduction_module_b(x, self.filter_multiplier, 1)

        for i in range(self.num_c_modules):
            x = inception_module_c(x, self.filter_multiplier, i)

        final_output = self.classifier(x)

        model = tf.keras.Model(inputs, final_output, name='DeNovoInceptionV4')
        if self.auxiliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1],
                                   name='DeNovoInceptionV4')

        return model
