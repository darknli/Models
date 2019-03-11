from data import *
from model import get_model_lm, get_model_text_classification
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os

bptt = 30


checkpointer = ModelCheckpoint(
        filepath=os.path.join('model', 'gru-.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

data = Data('data/train.txt')
train_data, train_label = data.get_data('data/train.txt', bptt)
print(train_data.shape, train_label.shape)
vocab_size = data.get_vocab_size()
print(vocab_size)
model = get_model_lm(vocab_size=vocab_size, bptt=bptt)

optimizer = Adam(lr=0.001, decay=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.output_shape)
model.fit(train_data, train_label, batch_size=32, epochs=100, callbacks=[checkpointer])
