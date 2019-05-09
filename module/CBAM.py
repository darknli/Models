from tensorflow.python.keras.layers import Add, Concatenate, \
    Conv2D, Multiply, Input, GlobalAveragePooling2D, GlobalMaxPool2D, Dense
from tensorflow.python.keras.models import Model
"""
the code write Convolutional Block Attention Module, and the paper url "https://arxiv.org/pdf/1807.06521.pdf"
"""
def channel_attention(input):
    avg_pool = GlobalAveragePooling2D()(input)
    max_pool = GlobalMaxPool2D()(input)
    pool = Add()([avg_pool, max_pool])
    net = Dense(pool.shape[-1], activation="relu")(pool)
    net = Dense(pool.shape[-1], activation="sigmoid")(net)
    output = Multiply()[input, net]
    return output


def spatial_attention(input):
    avg_pool = GlobalAveragePooling2D()(input)
    max_pool = GlobalMaxPool2D()(input)
    pool = Concatenate()([avg_pool, max_pool])
    net = Conv2D(1, (7, 7), (1, 1))(pool)
    output = Multiply()[input, net]
    return output


def CBAM(input_shape):
    feature_map = Input(shape=input_shape, name="CBAM_inputs")
    channel_choices = channel_attention(feature_map)
    return spatial_attention(channel_choices)

