import tensorflow as tf


def classifier(x, name='main', j=0):
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name=name + "_classifier_" + str(j))(x)
    return outputs


def multi_layer_perceptron(x, hidden_layer_size):
    return tf.keras.layers.Dense(hidden_layer_size, activation=tf.keras.activations.gelu)(x)


def conv_block(x, model_width, kernel, strides=(1, 1), padding="same", activation='relu', reverse=False):
    if not reverse:
        x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding,
                                   kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    if reverse:
        x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding,
                                   kernel_initializer="he_normal")(x)
    return x


def inception_module(inputs, filter_multiplier):
    branch1x1 = conv_block(inputs, 4 * filter_multiplier, (1, 1))

    branch3x3 = conv_block(inputs, 4 * filter_multiplier, (1, 1))
    branch3x3 = conv_block(branch3x3, 8 * filter_multiplier, (3, 3))

    branch5x5 = conv_block(inputs, 1 * filter_multiplier, (1, 1))
    branch5x5 = conv_block(branch5x5, 2 * filter_multiplier, (5, 5))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv_block(branch_pool, 2 * filter_multiplier, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return out


def residual_module(inputs, filter_multiplier, decrease_size=False, increase_depth=False):
    if decrease_size or increase_depth:
        shortcut = conv_block(inputs, 4 * filter_multiplier, (1, 1), strides=(2, 2) if decrease_size else (1, 1),
                              padding='valid',
                              activation='linear')
    else:
        shortcut = inputs

    x = conv_block(inputs, filter_multiplier, (3, 3))
    x = conv_block(x, 4 * filter_multiplier, (1, 1), strides=(2, 2) if decrease_size else (1, 1), padding='valid',
                   activation='linear')

    x = tf.keras.layers.add([shortcut, x])

    out = tf.keras.layers.Activation('relu')(x)
    return out


def dense_module(inputs, filter_multiplier):
    x = conv_block(inputs, filter_multiplier * 2, (1, 1), reverse=True)
    x = conv_block(x, filter_multiplier // 2, (3, 3), reverse=True)

    out = tf.keras.layers.concatenate([inputs, x], axis=-1)
    return out


def patch_extractor(inputs, patch_size):
    batch_size = tf.shape(inputs)[0]
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, (tf.shape(inputs)[1] // patch_size * tf.shape(inputs)[2] // patch_size),
                                   patch_dims])
    return patches


def patch_encoder(inputs, projection_dim):
    patches_embedding = tf.keras.layers.Dense(projection_dim)(inputs)

    positions = tf.range(start=0, limit=inputs.shape[1], delta=1)
    positions_embedding = tf.keras.layers.Embedding(input_dim=inputs.shape[1], output_dim=projection_dim)(
        positions)

    return patches_embedding + positions_embedding


def vision_transformer_module(inputs, num_heads, projection_dim):
    attention_branch = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(
        attention_branch, attention_branch)
    attention_branch = tf.keras.layers.Add()([attention, attention_branch])
    mlp_branch = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_branch)
    mlp_branch = multi_layer_perceptron(mlp_branch, projection_dim)
    return tf.keras.layers.Add()([mlp_branch, attention_branch])