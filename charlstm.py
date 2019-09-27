import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
import numpy as np
import pathlib

import time
import json



# Character Level Model ======================================================
# (If it looks silly, it's at least partly because it's adapted from a
#  word-level model I was fiddling with earlier.  Sorry.)

class CharLSTM(nn.Module):

    def __init__(self, charset, embedding_dim=256, hidden_dim=512, 
            num_layers=2, dropout=0.0):
        super(CharLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.charset = charset
        self.charset_size = len(charset)
        self.char_to_index = {char: i for i, char in enumerate(charset)}

        self.char_embeddings = nn.Embedding(self.charset_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers,dropout=self.dropout)
        self.hidden2chars = nn.Linear(hidden_dim, self.charset_size)

        self.device = torch.device("cpu")

    def init_hidden(self,batch_size=1):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (hidden, cell)

    def forward(self, sequence, hidden):
        batch_size = sequence.size(1)
        embeds = self.char_embeddings(sequence)
        lstm_out, hidden = self.lstm(
            embeds.view(len(sequence), batch_size, -1), hidden)
        char_space = self.hidden2chars(lstm_out.view(len(sequence)*batch_size,-1))
        char_scores = F.log_softmax(char_space, dim=1)
        return char_scores, hidden

    def set_device(self,device=torch.device("cpu")):
        self.device = device

    def gen_text(self, start_text, n_steps=200,hidden=None):
        ''' Generate text probabilistically, according to softmax output '''
        if not self.charset:
            raise RuntimeError("Must add charset to model in order to generate text")
        device = self.device
        seq = self.text_to_sequence(start_text)

        output = None
        input_seq = seq
        if hidden:
            h, c = hidden
        else:
            h, c = self.init_hidden(1)
        hidden = (h.to(device), c.to(device))
        for i in range(n_steps):
            output, hidden = self(input_seq, hidden)
            probabilities = output[-1,:]
            # Note: exp(logsoftmax) = softmax
            probabilities = np.exp(np.array(probabilities.detach().tolist()))
            # Apparently need to normalize slightly due to floating point error.
            probabilities = probabilities / np.sum(probabilities)

            # Choose next char according to probability distribution
            result = np.random.multinomial(1,probabilities)
            next_char_index = result.tolist().index(1)
            next_tensor = torch.tensor([[next_char_index]]).to(device)
            seq = torch.cat((seq, next_tensor))
            input_seq = next_tensor

        chars = [self.charset[i] for i in seq]
        return "".join(chars)

    def save_with_info(self, dir_path):
        pathlib.Path(dir_path).mkdir(parents=True,exist_ok=True)
        model_path = dir_path + "/model.bin"
        torch.save(self.state_dict(),model_path)

        args = {
            "charset_size": self.charset_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "charset": self.charset
        }
        s = json.dumps(args)
        with open(dir_path + "/model_info.json","w") as f:
            f.write(s)

    def text_to_sequence(self, text):
        ''' Convert text to a tensor of character indices '''
        seq = torch.LongTensor([self.char_to_index[c] for c in text])
        seq = seq.view(len(seq),1).to(self.device)
        return seq

    @classmethod
    def load_with_info(cls, dir_path):
        args = []
        with open(dir_path + "/model_info.json","r") as f:
            args = json.loads(f.read())
        model = cls(args["charset"],args["embedding_dim"],
            args["hidden_dim"],args["num_layers"],
            args["dropout"])
        model.load_state_dict(torch.load(dir_path + "/model.bin"))
        return model





