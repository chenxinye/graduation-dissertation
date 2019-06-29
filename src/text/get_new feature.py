#@author:chenxinye
#@2019.06.09

#see more,visit https://www.kaggle.com/chenxinye/apply-the-model-to-other-dataset

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import gc

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
def get_embedding():
    EMBEDDING_FILES = [
        'input/wordvec/crawl-300d-2M.vec',
        'input/wordvec/glove.840B.300d.txt'
    ]
    
    def embedding_build(path):
        embeddings_index = {}
        f = open(os.path.join(path),'r',encoding='utf-8')
        for line in f.readlines():
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError as reason:
                print('wrong!:\n',line)
                print(reason)
                pass
            embeddings_index[word]=coefs
        f.close()
        return embeddings_index
    
    embeddings_1 = embedding_build(EMBEDDING_FILES[0])
    embeddings_2 = embedding_build(EMBEDDING_FILES[1])
    embedding_word = dict(embeddings_1, **embeddings_2)
    del embeddings_1,embeddings_2
    import gc;gc.collect()
    print('Found %s word vectors.' % len(embedding_word))
    # Save
    np.save('embedding_word.npy', embedding_word)
    return embedding_word

#get_embedding()
gc.collect()
3#Load
embedding_word = np.load('embedding_word.npy').item()

def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):
    from tqdm import tqdm
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(),disable = not verbose):
        if lower:
            word = word.lower()
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(250,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(160, return_sequences=True))(x)
    at = Attention(250)(x)
    #x = Bidirectional(CuDNNLSTM(160, return_sequences=True))(x)
    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
        at
    ])
    
    hidden = add([hidden, Dense(960, activation='relu')(hidden)])
    hidden = add([hidden, Dense(960, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def preprocess(data):
    '''Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution'''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text
    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data

train = pd.read_csv('input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
traincol = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat','sexual_explicit']

def to_vec(data):
    data = preprocess(data)
    data = tokenizer.texts_to_sequences(data)
    data = sequence.pad_sequences(data, maxlen=250)
    return(data)
    
x_train = preprocess(train['comment_text'])
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[traincol ]

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=250)

word_index = tokenizer.word_index
embedding_matrix = build_embedding_matrix(word_index, embedding_word,max_features = 300)

checkpoint_predictions = []
weights = []

model = build_model(embedding_matrix, y_aux_train.shape[-1])
model.summary()

history_ = model.fit(x_train,
    [y_train,y_aux_train],
    batch_size=1000,
    epochs=10,
    verbose=2,callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * 0.1)])

text = pd.read_csv("text_combine.csv")
textvec = to_vec(text["combine_topic"])
predictions_text = model.predict(textvec, batch_size=2048)[1]

j = 0
for i in traincol:
    text[i] = predictions_text[:,j]
    j += 1 
    
text.to_csv('text_test.csv', index=False)

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

import matplotlib.pyplot as plt
plt.plot(history_.history['loss'])
plt.plot(history_.history['dense_3_loss'])
plt.plot(history_.history['dense_4_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'dense_3_loss','dense_4_loss'], loc='upper')
plt.ylim(0,0.5)
plt.show()
