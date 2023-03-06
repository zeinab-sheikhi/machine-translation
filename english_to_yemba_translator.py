from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import numpy as np
from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu

file = open('/Users/macbookpro/Python/TowardsDataScience/LanguageTranslatorWithKeras/data.txt')
text = file.read()

# Convert to List
lines = text.strip().split('\n')
pairs = [line.split(';') for line in lines]
pairs = np.array(pairs)


# fit a tokenizer
def create_toeknizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequence(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# define seq2seq model
def define_model(src_vocab, tar_vocab, source_steps, target_steps,
                 embedding_dim):
    model = Sequential()
    # encoder
    model.add(Embedding(src_vocab, embedding_dim, input_length=source_steps,
              mask_zero=True))
    model.add(LSTM(embedding_dim))
    model.add(RepeatVector(target_steps))
    # decoder
    model.add(LSTM(embedding_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    model.summary()
    return model


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# evaluate the model
def evaluate_model(model, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, yemba_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src,
                  raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


english_tokenizer = create_toeknizer(pairs[:, 1])
english_vocabulary_size = len(english_tokenizer.word_index) + 1
english_length = max_length(pairs[:, 1])

yemba_tokenizer = create_toeknizer(pairs[:, 0])
yemba_vocabualry_size = len(yemba_tokenizer.word_index) + 1
yemba_length = max_length(pairs[:, 0])


# shuffle data
dataset = np.array(pairs)
np.random.shuffle(dataset)
train, test = dataset[:40, :], dataset[40:, :]

# prepare training data
trainX = encode_sequence(english_tokenizer, english_length, train[:, 1])
trainY = encode_sequence(yemba_tokenizer, yemba_length, train[:, 0])
trainY = encode_output(trainY, yemba_vocabualry_size)


# prepare validation data
testX = encode_sequence(english_tokenizer, english_length, test[:, 1])
testY = encode_sequence(yemba_tokenizer, yemba_length, test[:, 0])
testY = encode_output(testY, yemba_vocabualry_size)

model = define_model(english_vocabulary_size, yemba_vocabualry_size,
                     english_length, yemba_length, 256)
checkpoint = ModelCheckpoint('model_en_yb.h5', monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
history = model.fit(trainX, trainY, epochs=1000, batch_size=16,
                    validation_data=(testX, testY),
                    callbacks=[checkpoint], verbose=2)
evaluate_model(model, testX, test)
