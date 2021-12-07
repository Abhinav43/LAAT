import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiGRU(nn.Module):
    """BiGRU (Bidirectional Gated Recurrent Unit)
    Args:
        embed_vecs (FloatTensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the GRU network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): Number of recurrent layers. Defaults to 1.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """
    def __init__(
        self,
        rnn_att  =0,
        rnn_dim=1024,
        
        rnn_layers=1,
        dropout=0.2,
        activation='tanh',
        **kwargs
    ):
        super(BiGRU, self).__init__()
        assert rnn_dim%2 == 0, """`rnn_dim` should be even."""

        # BiGRU
        emb_dim = 100
        self.rnn = nn.GRU(emb_dim, rnn_dim//2, rnn_layers,
                          bidirectional=True, batch_first=True)

        # context vectors for computing attention
        self.U = nn.Linear(rnn_dim, 4000)
        xavier_uniform_(self.U.weight)
        
        
        self.rnn_att      = rnn_att
        

    def forward(self, x, lengths):
       

        packed_inputs = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, _ = self.rnn(packed_inputs)
        x = pad_packed_sequence(x)[0]
        x = x.permute(1, 0, 2)

        # Apply per-label attention
        # 8 x 4000 x 1024
        # 8 x 4000, 1024 (linear) * 8 x 1024 x 4000 
        # 8 x 4000 x 4000 * 8 x 4000 x 1024
        
        if self.rnn_att:
            x = torch.tanh(x)
            alpha = torch.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

            # Document representations are weighted sums using the attention
            m = alpha.matmul(x) # (batch_size, num_classes, rnn_dim)
            x = m

        return x
