from keras.layers import GRU, Embedding, Input, Bidirectional, Dense, TimeDistributed, Flatten
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam
class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_model_text_classification(bptt, vocab_size, num_classes):
    inputs = Input(shape=(bptt,), dtype='float64')
    embed = Embedding(vocab_size + 1, 300)(inputs)
    gru = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(embed)
    attention = Attention(gru)
    print(attention.output_shape)
    output = Dense(num_classes, activation='softmax')(attention)
    model = Model(inputs, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model


def get_model_lm(vocab_size, bptt=30):
    inputs = Input(shape=(bptt, ), dtype='float64')
    embed = Embedding(vocab_size, 300, input_length=bptt)(inputs)
    gru = GRU(100, dropout=0.2, return_sequences=True)(embed)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(gru)
    output = Flatten()(output)
    model = Model(inputs, output)
    # model = Sequential()
    # model.add(Embedding(vocab_size, 10, input_length=1))
    # model.add(GRU(50, return_sequences=True))
    # model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    return model