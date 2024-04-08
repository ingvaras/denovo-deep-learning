import tensorflow as tf


def conv_block(x, model_width, kernel, strides=(1, 1), padding="same"):
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def inception_module_a(inputs, filter_multiplier, i):
    branch1x1 = conv_block(inputs, 3 * filter_multiplier, (1, 1))

    branch5x5 = conv_block(inputs, 2 * filter_multiplier, (1, 1))
    branch5x5 = conv_block(branch5x5, 3 * filter_multiplier, (5, 5))

    branch3x3 = conv_block(inputs, 2 * filter_multiplier, (1, 1))
    branch3x3 = conv_block(branch3x3, 3 * filter_multiplier, (3, 3))
    branch3x3 = conv_block(branch3x3, 3 * filter_multiplier, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv_block(branch_pool, 3 * filter_multiplier, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=-1,
                                      name='Inception_Module_A' + str(i))
    return out


def inception_module_b(inputs, filter_multiplier, i):
    branch1x1 = conv_block(inputs, 12 * filter_multiplier, (1, 1))

    branch7x7 = conv_block(inputs, 6 * filter_multiplier, (1, 1))
    branch7x7 = conv_block(branch7x7, 8 * filter_multiplier, (1, 7))
    branch7x7 = conv_block(branch7x7, 8 * filter_multiplier, (7, 1))

    branch7x7_2 = conv_block(inputs, 6 * filter_multiplier, 1)
    branch7x7_2 = conv_block(branch7x7_2, 7 * filter_multiplier, (1, 7))
    branch7x7_2 = conv_block(branch7x7_2, 7 * filter_multiplier, (7, 1))
    branch7x7_2 = conv_block(branch7x7_2, 8 * filter_multiplier, (1, 7))
    branch7x7_2 = conv_block(branch7x7_2, 8 * filter_multiplier, (7, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv_block(branch_pool, 4 * filter_multiplier, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7_2, branch_pool], axis=-1,
                                      name='Inception_Module_B' + str(i))
    return out


def inception_module_c(inputs, filter_multiplier, i):
    branch1x1 = conv_block(inputs, 8 * filter_multiplier, (1, 1))

    branch3x3 = conv_block(inputs, 12 * filter_multiplier, (1, 1))
    branch3x3_a = conv_block(branch3x3, 16 * filter_multiplier, (1, 3))
    branch3x3_b = conv_block(branch3x3, 16 * filter_multiplier, (3, 1))

    branch3x3_2 = conv_block(inputs, 12 * filter_multiplier, (1, 1))
    branch3x3_2 = conv_block(branch3x3_2, 16 * filter_multiplier, (1, 3))
    branch3x3_2 = conv_block(branch3x3_2, 16 * filter_multiplier, (3, 1))
    branch3x3_2a = conv_block(branch3x3_2, 16 * filter_multiplier, (1, 3))
    branch3x3_2b = conv_block(branch3x3_2, 16 * filter_multiplier, (3, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv_block(branch_pool, 8 * filter_multiplier, (1, 1))

    out = tf.keras.layers.concatenate(
        [branch1x1, branch3x3_a, branch3x3_b, branch3x3_2a, branch3x3_2b, branch_pool], axis=-1,
        name='Inception_Module_C' + str(i))
    return out


def inception_reduction_module_a(inputs, filter_multiplier, i):
    branch3x3 = conv_block(inputs, 2 * filter_multiplier, (1, 1))
    branch3x3 = conv_block(branch3x3, 12 * filter_multiplier, (3, 3), strides=(2, 2))

    branch3x3_2 = conv_block(inputs, 6 * filter_multiplier, (1, 1))
    branch3x3_2 = conv_block(branch3x3_2, 7 * filter_multiplier, (3, 3))
    branch3x3_2 = conv_block(branch3x3_2, 8 * filter_multiplier, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3_2, branch_pool], axis=-1,
                                      name='Inception_Reduction_Module_A_' + str(i))
    return out


def inception_reduction_module_b(inputs, filter_multiplier, i):
    branch3x3 = conv_block(inputs, 6 * filter_multiplier, (1, 1))
    branch3x3 = conv_block(branch3x3, 6 * filter_multiplier, (3, 3), strides=(2, 2))

    branch3x3_2 = conv_block(inputs, 8 * filter_multiplier, (1, 1))
    branch3x3_2 = conv_block(branch3x3_2, 10 * filter_multiplier, (1, 7))
    branch3x3_2 = conv_block(branch3x3_2, 10 * filter_multiplier, (7, 1))
    branch3x3_2 = conv_block(branch3x3_2, 10 * filter_multiplier, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3_2, branch_pool], axis=-1,
                                      name='Inception_Reduction_Module_B_' + str(i))
    return out
