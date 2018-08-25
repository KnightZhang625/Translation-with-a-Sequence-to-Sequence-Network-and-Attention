from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SOS_token = 0
EOS_token = 1

class Lang(object):
    def __init__(self, name):
        self.name = name 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'SOS', 1:'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class ReadData(object):
    def __init__(self, lang1, lang2, reverse=False):
        self.lang1 = lang1
        self.lang2 = lang2
        self.reverse = reverse

        self.MAX_LENGTH = 10
        self.eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re ")

    def unicodeToAscii(self, sentence):
        return ''.join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')

    def normalizeString(self, sentence):
        sentence = self.unicodeToAscii(sentence.lower().strip())
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        return sentence

    def readLangs(self):
        lines = open('data/%s-%s.txt'%(self.lang1, self.lang2), encoding='utf-8').read().strip().split('\n')
        pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]

        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.lang2)
            output_lang = Lang(self.lang1)
        else:
            input_lang = Lang(self.lang1)
            output_lang = Lang(self.lang2)
        return input_lang, output_lang, pairs

    def prepareData(self):
        input_lang, output_lang, pairs = self.readLangs()
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs

    def filterPair(self, p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and \
            len(p[1].split(' ')) < self.MAX_LENGTH and \
            p[1].startswith(self.eng_prefixes)


    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.hidden = self.initHidden()

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = embedded.view(input.size()[0], 1, -1)
        output, hidden = self.gru(embedded, self.hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, 1)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)     # 1 * 1 * hidden_size

        embedded = self.dropout(embedded)

        # 假设现在是decoder的t=1时刻
        # 我们要把S<0>(decoder的前一隐藏值)和encoder的每一个hidden state放进NN中
        # S<0> : 1 * hidden_size   所有hidden_state : len(句子) * hidden_state
        # 所以我们要扩展S<0>, i.e. hidden
        # hidden : 1 * 1 * hidden_size, hidden[0] : 1 * hidden_size
        hidden_expand = hidden[0].expand(encoder_outputs.size()[0], self.hidden_size)
        # 这一步将S<0>, encoder所有hidden_state合并, 变为 len(句子) * hidden_size 
        hidden_encoder_combine = torch.cat((hidden_expand, encoder_outputs), 1)

        # 下一步把S<0>, encoder所有hidden_state送到NN, 并算softmax, 记住, 我们假设现在位于decoder的t=1时刻, 所以decoder的前一时刻是S<0>
        attn_weights = F.softmax(self.attn(hidden_encoder_combine), dim=1)  # len(句子) * 1

        # 下一步用各自的a<t,t'>乘以encoder每一个hidden_state, 矩阵相乘, 直接完成累加
        c = torch.mm(attn_weights.t(), encoder_outputs)     # (1 * len(句子)) dot (len(句子) * hidden_size) = 1 * hidden_state
    
        # 下一步将输入单词和attention matrix结合起来, 1 * (hidden_size * 2)
        input_t = torch.cat((embedded[0], c), 1)
        input_combine = self.attn_combine(input_t).view(1, 1, -1)   # 变为三维, 因为GRU需要batch维度

        output, hidden = self.gru(input_combine, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

teacher_forcing_ratio = 0.5

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, 256)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei])

        encoder_outputs[ei] = encoder_output[0]

    decoder_input = torch.tensor([[SOS_token]])     # 1*1

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(0)  # detach from history as input
            # print(decoder_input.size())
            # sys.exit()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(pairs, encoder, decoder, n_iters, print_every=1, plot_every=100, learning_rate=0.01):
    # start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(plot_loss_total)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)

if __name__ == '__main__':

    data = ReadData('eng', 'fra', True)
    input_lang, output_lang, pairs = data.prepareData()
    # print(random.choice(pairs))

    hidden_size = 256

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)

    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout=0.1)

    trainIters(pairs, encoder1, attn_decoder1, 75000, print_every=5000)








