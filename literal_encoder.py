import numpy as np
import gc
import torch
import torch.nn as nn
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


class AutoEncoder(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def generate_unlisted_word2vec(word2vec, literal_list, vector_dimension):
    unlisted_words = []
    for literal in literal_list:
        words = literal.split(' ')
        for word in words:
            if word not in word2vec:
                unlisted_words.append(word)

    character_vectors = {}
    alphabet = ''
    ch_num = {}
    for word in unlisted_words:
        for ch in word:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]
    print(alphabet)
    print('len(alphabet):', len(alphabet), '\n')
    char_sequences = [list(word) for word in unlisted_words]
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    for ch in alphabet:
        assert ch in model
        character_vectors[ch] = model[ch]

    word2vec_new = {}
    for word in unlisted_words:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            word2vec_new[word] = vec / len(word)

    word2vec.update(word2vec_new)
    return word2vec


class LiteralEncoder:
    def __init__(self, literal_list, word2vec, args, word2vec_dimension):
        self.args = args
        self.literal_list = literal_list
        self.word2vec = generate_unlisted_word2vec(word2vec, literal_list, word2vec_dimension)
        self.tokens_max_len = self.args.literal_len
        self.word2vec_dimension = word2vec_dimension

        literal_vector_list = []
        for literal in self.literal_list:
            vectors = np.zeros((self.tokens_max_len, self.word2vec_dimension), dtype=np.float32)
            words = literal.split(' ')
            for i in range(min(self.tokens_max_len, len(words))):
                if words[i] in self.word2vec:
                    vectors[i] = self.word2vec[words[i]]
            literal_vector_list.append(vectors)
        assert len(literal_list) == len(literal_vector_list)
        autoencoder = AutoEncoder(literal_vector_list, self.args)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
        loss_func = nn.MSELoss()
        batch_size = self.args.batch_size
        num_batch = len(self.literal_vector_list) // batch_size + 1
        batches = list()
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(self.word_vec_list[i * batch_size:])
            else:
                batches.append(self.word_vec_list[i * batch_size:(i + 1) * batch_size])

        for i in range(self.args.encoder_epoch):
            for i in range(num_batch):
                encoded, decoded = autoencoder(batches[i])
                loss = loss_func(decoded,batches[i])  # mean square error
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()
                print('Epoch: ', i, '| train loss: %.4f' % loss.data.numpy())
        self.encoded_literal_vector = []
        for i in range(num_batch):
            encoded , decoded = autoencoder(batches[i])
            self.encoded_literal_vector.append(encoded)

        self.encoded_literal_vector = autoencoder.encoder_multi_batches(literal_vector_list)



