import tensorflow as tf

def channel_attention(x, reduction=8):
    """Channel attention mechanism"""
    channels = x.shape[-1]
    # Global average pooling
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    # FC layers
    fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')(gap)
    fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')(fc1)
    # Reshape and multiply
    fc2 = tf.keras.layers.Reshape((1, 1, channels))(fc2)
    return x * fc2

def build_improved_model(img_height=128, img_width=224):
    """Improved U-Net with attention and better skip connections"""
    inputs = tf.keras.layers.Input(shape=[img_height, img_width, 6])

    # Encoder with BatchNorm
    e1 = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1 = tf.keras.layers.ReLU()(e1)
    e1 = tf.keras.layers.Conv2D(64, 3, padding='same')(e1)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1 = tf.keras.layers.ReLU()(e1)
    p1 = tf.keras.layers.MaxPooling2D(2)(e1)

    e2 = tf.keras.layers.Conv2D(128, 3, padding='same')(p1)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2 = tf.keras.layers.ReLU()(e2)
    e2 = tf.keras.layers.Conv2D(128, 3, padding='same')(e2)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2 = tf.keras.layers.ReLU()(e2)
    p2 = tf.keras.layers.MaxPooling2D(2)(e2)

    e3 = tf.keras.layers.Conv2D(256, 3, padding='same')(p2)
    e3 = tf.keras.layers.BatchNormalization()(e3)
    e3 = tf.keras.layers.ReLU()(e3)
    e3 = tf.keras.layers.Conv2D(256, 3, padding='same')(e3)
    e3 = tf.keras.layers.BatchNormalization()(e3)
    e3 = tf.keras.layers.ReLU()(e3)
    p3 = tf.keras.layers.MaxPooling2D(2)(e3)

    e4 = tf.keras.layers.Conv2D(512, 3, padding='same')(p3)
    e4 = tf.keras.layers.BatchNormalization()(e4)
    e4 = tf.keras.layers.ReLU()(e4)
    e4 = tf.keras.layers.Conv2D(512, 3, padding='same')(e4)
    e4 = tf.keras.layers.BatchNormalization()(e4)
    e4 = tf.keras.layers.ReLU()(e4)
    p4 = tf.keras.layers.MaxPooling2D(2)(e4)

    # Bottleneck with attention
    b = tf.keras.layers.Conv2D(1024, 3, padding='same')(p4)
    b = tf.keras.layers.BatchNormalization()(b)
    b = tf.keras.layers.ReLU()(b)
    b = tf.keras.layers.Conv2D(1024, 3, padding='same')(b)
    b = tf.keras.layers.BatchNormalization()(b)
    b = tf.keras.layers.ReLU()(b)
    b = channel_attention(b)

    # Decoder with skip connections and attention
    d4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same')(b)
    e4_att = channel_attention(e4)
    d4 = tf.keras.layers.concatenate([d4, e4_att])
    d4 = tf.keras.layers.Conv2D(512, 3, padding='same')(d4)
    d4 = tf.keras.layers.BatchNormalization()(d4)
    d4 = tf.keras.layers.ReLU()(d4)
    d4 = tf.keras.layers.Conv2D(512, 3, padding='same')(d4)
    d4 = tf.keras.layers.BatchNormalization()(d4)
    d4 = tf.keras.layers.ReLU()(d4)

    d3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(d4)
    e3_att = channel_attention(e3)
    d3 = tf.keras.layers.concatenate([d3, e3_att])
    d3 = tf.keras.layers.Conv2D(256, 3, padding='same')(d3)
    d3 = tf.keras.layers.BatchNormalization()(d3)
    d3 = tf.keras.layers.ReLU()(d3)
    d3 = tf.keras.layers.Conv2D(256, 3, padding='same')(d3)
    d3 = tf.keras.layers.BatchNormalization()(d3)
    d3 = tf.keras.layers.ReLU()(d3)

    d2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d3)
    e2_att = channel_attention(e2)
    d2 = tf.keras.layers.concatenate([d2, e2_att])
    d2 = tf.keras.layers.Conv2D(128, 3, padding='same')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    d2 = tf.keras.layers.ReLU()(d2)
    d2 = tf.keras.layers.Conv2D(128, 3, padding='same')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    d2 = tf.keras.layers.ReLU()(d2)

    d1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d2)
    e1_att = channel_attention(e1)
    d1 = tf.keras.layers.concatenate([d1, e1_att])
    d1 = tf.keras.layers.Conv2D(64, 3, padding='same')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)
    d1 = tf.keras.layers.Conv2D(64, 3, padding='same')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)

    # Output with sigmoid (direct prediction, no residual)
    outputs = tf.keras.layers.Conv2D(3, 1, activation='sigmoid', padding='same')(d1)

    model = tf.keras.Model(inputs, outputs)
    return model