from tensorflow.python.keras.layers import Add, Concatenate, \
    Conv2D, Multiply, Input, GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Lambda
from tensorflow.python.keras.models import Model
"""
the code write Convolutional Block Attention Module, and the paper url "https://arxiv.org/pdf/1807.06521.pdf"
"""
def broadcast_multiply(input1, input2):
    return input1 * input2


def channel_attention(input):
    avg_pool = GlobalAveragePooling2D()(input)
    max_pool = GlobalMaxPool2D()(input)
    pool = Add()([avg_pool, max_pool])
    net = Dense(pool.shape[-1], activation="relu")(pool)
    net = Dense(pool.shape[-1], activation="sigmoid")(net)
    net = Reshape((1, 1, pool.shape[-1]))(net)
    # output = Multiply()[input, net]
    output = Lambda(broadcast_multiply, arguments={'input2': net})(input)
    return output


def spatial_attention(input):
    avg_pool = Lambda(k.mean, arguments={'axis': -1}, name="spatial_attention_avg_pool")(input)
    avg_pool = Lambda(k.expand_dims, arguments={'axis': -1})(avg_pool)
    max_pool = Lambda(k.max, arguments={'axis': -1})(input)
    max_pool = Lambda(k.expand_dims, arguments={'axis': -1})(max_pool)
    pool = Concatenate(axis=-1)([avg_pool, max_pool])
    net = Conv2D(1, (3, 3), (1, 1), padding="same", activation="sigmoid", name="spatial_attention_conv")(pool)
    output = Lambda(broadcast_multiply, arguments={'input2': net})(input)
    return output


def CBAM(input):
    channel_choices = channel_attention(input)
    return spatial_attention(channel_choices)