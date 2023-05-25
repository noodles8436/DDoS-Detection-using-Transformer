import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM, GRU, GlobalMaxPooling1D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from DatasetLoader import Dataset

loss_fn = BinaryCrossentropy()
early_stop = EarlyStopping(monitor='val_loss', patience=0, restore_best_weights=True)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, *args, **kwargs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


def TransModel(T, F) -> Model:
    inputs = Input(shape=(T, F))

    x = TransformerBlock(embed_dim=F, num_heads=3, ff_dim=F)(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=1e-3, decay=1e-4), loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    model.summary()

    return model


def LSTMModel(T, F) -> Model:
    inputs = Input(shape=(T, F))

    x = LSTM(units=9, return_sequences=True)(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=1e-3, decay=1e-4), loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    model.summary()

    return model


def GRUModel(T, F) -> Model:
    inputs = Input(shape=(T, F))

    x = GRU(units=9, return_sequences=True)(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=1e-3, decay=1e-4), loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    model.summary()

    return model


if __name__ == "__main__":
    _batch = 32
    learning_rate = 1e-3
    loss_fn = BinaryCrossentropy()

    # model = TransModel(3, 9)
    # model = LSTMModel(3, 9)
    model = GRUModel(3, 9)

    train_gen = Dataset(_batch, trainType=0)
    val_gen = Dataset(_batch, trainType=1)
    test_gen = Dataset(_batch, trainType=2)

    model.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=[early_stop])
    model.evaluate(test_gen)
