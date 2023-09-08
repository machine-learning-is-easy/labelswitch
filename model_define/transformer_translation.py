# codes from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""
python -m spacy download en
python -m spacy download de
"""
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from loss.partial_loss import LabelWiseSignificanceCrossEntropy
import io
import argparse
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR, ReduceLROnPlateau

torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--loss_function', '-lf', dest='lf', default="MASKEDLABEL", help='lossfunction', required=False)
parser.add_argument('--lr', '-lr', dest='lr', type=float, default=1e-2, help='learning rate')
args = parser.parse_args()

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    # return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab_dict = {'<unk>': 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for key in counter:
        if key.lower() not in vocab_dict:
            vocab_dict[key.lower()] = len(vocab_dict)
    return Vocab(vocab_dict)

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        try:
            de_tensor_ = torch.tensor([de_vocab.vocab[token.lower()] for token in de_tokenizer(raw_de)],
                                      dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab.vocab[token.lower()] for token in en_tokenizer(raw_en)],
                                      dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        except Exception as ex:
            continue
    return data

train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)


"""DataLoader
The last torch specific feature we’ll use is the DataLoader, which is easy to use since it takes the data as its first argument. Specifically, as the docs say: DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset. The DataLoader supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.

Please pay attention to collate_fn (optional) that merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.

"""

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


"""
Defining our nn.Module and Optimizer
That’s mostly it from a torchtext perspecive: with the dataset built and the iterator defined, the rest of this tutorial simply defines our model as an nn.Module, along with an Optimizer, and then trains it.

Our model specifically, follows the architecture described here (you can find a significantly more commented version here).

Note: this model is just an example model that can be used for language translation; we choose it because it is a standard model for the task, not because it is the recommended model to use for translation. As you’re likely aware, state-of-the-art models are currently based on Transformers; you can see PyTorch’s capabilities for implementing Transformer layers here; and in particular, the “attention” used in the model below is different from the multi-headed self-attention present in a transformer model.
"""

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
def define_opt(model):
    return optim.Adam(model.parameters(), lr=args.lr)

optimizer = define_opt(model)
# lambda_ = lambda epoch : (100 - epoch) / 100

def define_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer)
scheduler = define_scheduler(optimizer)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

"""
define loss function
"""

# PAD_IDX = en_vocab.stoi['<pad>']
PAD_IDX = en_vocab.vocab['<pad>']

if args.lf == "CROSSENTROPY":
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
elif args.lf == 'LABELSMOOTHING':
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
elif args.lf == "LWSCE":
    criterion = LabelWiseSignificanceCrossEntropy(ignore_index=PAD_IDX, alpha=0.2, num_class=OUTPUT_DIM, input_type="log", device=device)
else:
    raise Exception("Unsupport loss function of {}".format(args.lf))


"""
train model
"""

import math
import time


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        total = 0
        acc = 0
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            output_digit = torch.argmax(output, dim=-1)
            # ignore pad index
            not_pad = trg != PAD_IDX
            total += torch.sum(not_pad).item()
            acc += torch.sum(torch.logical_and(not_pad, output_digit == trg)).item()
    acc = acc/total
    return epoch_loss / len(iterator), acc


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')
import os
current_folder = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
test_acc = []
for _ in range(10):  # run multiple times
    file_path = os.path.join(current_folder, "result/{}_result/{}.csv".format(args.lf, _))
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    result = []
    for epoch in range(N_EPOCHS):

        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss, acc = evaluate(model, valid_iter, criterion)
        scheduler.step(metrics=acc)

        print(f'run time: {_} and epoch: {epoch}', f'Train Loss: {train_loss:.3f}', f' accuracy: {acc:.3f}')
        result.append([epoch, round(train_loss,3), round(acc, 3)])

    pd.DataFrame(result, columns=["epoch", "train_loss", "accuracy"]).to_csv(file_path)

    test_loss, acc = evaluate(model, test_iter, criterion)

    test_acc.append(acc)
    # redefine the model and optimizer
    del model
    del optimizer
    del scheduler
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    optimizer = define_opt(model)
    scheduler = define_scheduler(optimizer)


    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print(f'| Test acc: {acc:.3f}')

# save test accuracy to the folder
test_acc_file_path = os.path.join(current_folder, "result/{}_result/test_acc.csv".format(args.lf))
pd.DataFrame(test_acc, columns=["accuracy"]).to_csv(test_acc_file_path)
