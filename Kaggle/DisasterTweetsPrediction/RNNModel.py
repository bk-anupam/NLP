import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, vocab_size, num_layers, is_bidirect, emb_dim, hidden_dim, out_dim, 
                drop_prob=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim        
        self.output_dim = out_dim        
        # Embedding layer
        self.emb_layer = nn.Embedding(self.vocab_size, emb_dim)
        # LSTM Layer        
        self.lstm_layer = nn.LSTM(
                        input_size=emb_dim, hidden_size=hidden_dim, batch_first=True, 
                        bidirectional=is_bidirect, num_layers=num_layers, dropout=drop_prob
                        )
        self.dropout = nn.Dropout(p = drop_prob)                        
        
        # If the RNN is bidirectional `num_directions` should be 2, else it should be 1.        
        if not is_bidirect:
            self.num_directions = 1
            # The linear layer is for making predictions 
            # input to linear output layer is of shape num_steps, batch_size, num_hiddens
            # and output is of shape num_steps, batch_size, vocab_size
            # Wya is of shape (vocab_size, num_hiddens), a_out is of shape (num_hiddens, 1)
            # yt_pred = np.dot(Wya, a_out) + b is of shape (vocab_size, 1)
            # replace 1 with m (batch_size) and add num_steps as the first dimension to have
            # vectorized form of the output
            self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        else:       
            self.num_directions = 2     
            self.linear = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # The activation layer which converts output to 0 or 1            
        self.act = nn.Sigmoid()            

    def forward(self, inputs, input_lengths, state):        
        # inputs is of shape batch_size, num_steps(sequence length which is the length of
        # longest text sequence). Each row of inputs is 1d LongTensor array of length 
        # num_steps containing word index. Using the embedding layer we want to convert
        # each word index to its corresponding word vector of dimension emb_dim
        batch_size = inputs.size(0)
        num_steps = inputs.size(1)        
        # embeds is of shape batch_size * num_steps * emb_dim and is the input to lstm layer
        embeds = self.emb_layer(inputs)        
        # pack_padded_sequence before feeding into LSTM. This is required so pytorch knows
        # which elements of the sequence are padded ones and ignore them in computation.
        # This step is done only after the embedding step
        embeds_pack = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        # lstm_out is of shape batch_size * num_steps * hidden_size and contains the output
        # features (h_t) from the last layer of LSTM for each t
        # h_n is of shape num_layers * batch_size * hidden_size and contains the final hidden 
        # state for each element in the batch i.e. hidden state at t_end
        # same for c_n as h_n except that it is the final cell state
        lstm_out_pack, (h_n, c_n) = self.lstm_layer(embeds_pack)
        # unpack the output
        lstm_out, lstm_out_len = pack_padded_sequence(lstm_out_pack, batch_first=True)
        # tensor flattening works only if tensor is contiguous
        # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        # flatten lstm_out from 3d to 2d with shape (batch_size * num_steps), hidden_dim)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)        
        # regularize lstm output by applying dropout
        out = self.dropout(lstm_out)        
        # The the output Y of fully connected rnn layer has the shape of 
        # (`num_steps` * `batch_size`, `num_hiddens`). This Y is then fed as input to the 
        # output fully connected linear layer which produces the prediction in the output shape of 
        # (`num_steps` * `batch_size`, `output_dim`).        
        output = self.linear(out)
        # reshape output to batch_size, num_steps, output_dim
        output = output.view(batch_size, -1, self.output_dim)
        # reshape output again to batch_size, output_dim. The last element of middle dimension
        # i.e. num_steps is taken i.e. for each item in the batch the output is the hidden state
        # from the last layer of LSTM for t = t_end
        output = output[:, -1, :]
        output = self.act(output)
        return output, (h_n, c_n)

    def init_state(self, device, batch_size=1):
        """ Initialize the hidden state i.e. initialize all the neurons in all the hidden layers 
        to zero"""
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.hidden_dim), device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states (h0, c0). h0 = initial
            # hidden state for each element in the batch, c0 = initial cell state
            # for each element in the batch
            return (torch.zeros((self.num_directions * self.num_layers,
                                 batch_size, self.hidden_dim), device=device),
                    torch.zeros((self.num_directions * self.num_layers,
                                 batch_size, self.hidden_dim), device=device))