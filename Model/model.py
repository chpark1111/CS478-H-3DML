import torch
from torch.functional import Tensor
import torch.nn  as nn
import torch.nn.functional as F
from typing import List

class Encoder(nn.Module):
    def __init__(self, enc_drop=0.2):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.max_pool = nn.MaxPool2d(kernel_size  = 2)
        self.drop = nn.Dropout(enc_drop)

    def forward(self, x):
        """
        Input: (batch_size, in_ch, 64, 64)
        Outputs: (batch_size, out_ch*8*8)
        """
        y = self.max_pool(self.drop(F.relu(self.conv1(x))))
        y = self.max_pool(self.drop(F.relu(self.conv2(y))))
        y = self.max_pool(self.drop(F.relu(self.conv3(y))))
        y = y.reshape(y.shape[0], -1)
        return y

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, mode=1, num_layers=1, 
                    dropout=0.5, num_draws=400):
        super(Decoder, self).__init__()
        """
        :param input_size: input_size (CNN feature size) to rnn
        :param hidden_size: rnn hidden size
        :param mode: Mode of training, RNN, BDRNN or something else
        :param num_layers: Number of layers to rnn
        :param dropout: dropout
        :param num_draws: Total number of tokens present in the dataset or total number 
                    of operations to be predicted + a stop symbol = 400 in 2d
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_layers = num_layers
        self.num_draws = num_draws

        self.op_embed_size = 128

        self.rnn = nn.GRU(input_size=input_size + self.op_embed_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=False)
        self.op_ebmed_fc = nn.Linear(self.num_draws+1, self.op_embed_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_draws)
        self.drop = nn.Dropout(dropout)

        self.log_softmax = nn.LogSoftmax(dim = 2)

    def forward(self, x):
        """
        Input: (batch_size, seq_len, num_draws + 1), (batch_size, seq_len, self.input)
        Outputs: (seq_len, batch_size, num_draws)
        """
        prev_op, enc_f = x
        ebd_op = F.relu(self.op_ebmed_fc(prev_op))

        inp = torch.cat((self.drop(enc_f), ebd_op), 2)
        inp = inp.transpose(0, 1)
        h, _ = self.rnn(inp)
        
        y = F.relu(self.fc1(self.drop(h)))
        y = self.log_softmax(self.fc2(self.drop(y)))
        return y

class CSGmodel(nn.Module):
    def __init__(self, input_size, hidden_size, mode=1, num_layers=1, 
                    enc_drop=0.2, dropout=0.5, num_draws=400, canvas_shape=[64, 64]):
        super(CSGmodel, self).__init__()
        self.encoder = Encoder(enc_drop)
        self.decoder = Decoder(input_size, hidden_size, mode, num_layers, dropout, num_draws)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_layers = num_layers
        self.num_draws = num_draws
        self.canvas_shape = canvas_shape

        
    def forward(self, x: List):
        """
        Input: (pg_len+1, batch_size, 1(top of stack), 64, 64), (batch_size, pg_len+2, num_draws+1), (pg_len)
        Outputs: (batch_size, pg_len, num_draws)
        """
        
        if self.mode==1:
            '''
            For supervised training: goes to decoder forward
            
            encoder input
            Input: (batch_size, in_ch, 64, 64)
            decoder input
            Input: (batch_size, seq_len, num_draws + 1), (batch_size, seq_len, self.input_size)
            
            '''

            data, in_op, pg_len = x
            batch_size = data.size()[1]

            enc_f = self.encoder(data[-1, :, :, :, :])
            assert enc_f.shape == (batch_size, 2048)

            enc_f = torch.unsqueeze(enc_f, 1)
            enc_f = enc_f.repeat(1, pg_len+1, 1)

            y = self.decoder(in_op[:, :-1, :], enc_f)
            y = y.transpose(0, 1)
            return y
    
    def test(self, x):
        """
        Input: (pg_len+1, batch_size, 1(top of stack), 64, 64), (batch_size, pg_len+2, num_draws+1), (pg_len)
        Outputs: (batch_size, pg_len, num_draws)
        """

        if self.mode==1:
            data, in_op, pg_len = x
            batch_size = data.size()[1]

            enc_f = self.encoder(data[-1, :, :, :, :])
            assert enc_f.shape == (batch_size, 2048)

            enc_f = torch.unsqueeze(enc_f, 1)
            output = []
            last_out = in_op[:, 0:1, :]

            for i in range(pg_len):
                y = self.decoder(last_out, enc_f)
                y = torch.squeeze(y)
                output.append(y)

                next_op = torch.max(y, 1, keepdim=True)[1]
                last_out = (torch.zeros((batch_size, self.num_draws + 1)).scatter_(1, next_op, 1.0)).cuda()

            return output

    def beam_search(self, x, k=1):
        data, input_op = x
        return x