# # -*- coding: utf-8 -*-
# """
#     Word-based RNN model for text classification
#     @author: Thanh Vu <thanh.vu@csiro.au>
#     @date created: 07/03/2019
#     @date last modified: 19/08/2020
# """

# from torch.autograd import Variable
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from src.models.attentions.util import *
# from src.models.embeddings.util import *
# from src.data_helpers.vocab import Vocab, device


# class RNN(nn.Module):
#     def __init__(self, vocab: Vocab,
#                  args):
#         """

#         :param vocab: Vocab
#             The vocabulary normally built on the training data
#         :param args:
#             mode: rand/static/non-static/multichannel the mode of initialising embeddings
#             hidden_size: (int) The size of the hidden layer
#             n_layers: (int) The number of hidden layers
#             bidirectional: (bool) Whether or not using bidirectional connection
#             dropout: (float) The dropout parameter for RNN (GRU or LSTM)
#         """

#         super(RNN, self).__init__()
#         self.vocab_size = vocab.n_words()
#         self.vocab = vocab
#         self.args = args
#         self.use_last_hidden_state = args.use_last_hidden_state
#         self.mode = args.mode
#         self.n_layers = args.n_layers
#         self.hidden_size = args.hidden_size
#         self.bidirectional = bool(args.bidirectional)
#         self.n_directions = int(self.bidirectional) + 1
#         self.attention_mode = args.attention_mode
#         self.output_size = self.hidden_size * self.n_directions
#         self.rnn_model = args.rnn_model

#         self.dropout = args.dropout
#         self.embedding = init_embedding_layer(args, vocab)

#         if self.rnn_model.lower() == "gru":
#             self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
#                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
#         else:
#             self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
#                                bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

#         self.use_dropout = args.dropout > 0
#         self.dropout = nn.Dropout(args.dropout)
#         init_attention_layer(self)

#     def init_hidden(self,
#                     batch_size: int = 1) -> Variable:
#         """
#         Initialise the hidden layer
#         :param batch_size: int
#             The batch size
#         :return: Variable
#             The initialised hidden layer
#         """
#         # [(n_layers x n_directions) x batch_size x hidden_size]
#         h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(device)
#         c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(device)
#         if self.rnn_model.lower() == "gru":
#             return h
#         return h, c

#     def forward(self,
#                 batch_data: torch.LongTensor,
#                 lengths: torch.LongTensor) -> tuple:
#         """

#         :param batch_data: torch.LongTensor
#             [batch_size x max_seq_len]
#         :param lengths: torch.LongTensor
#             [batch_size x 1]
#         :return: output [batch_size x n_classes]
#             attention_weights
#         """

#         batch_size = batch_data.size()[0]
#         hidden = self.init_hidden(batch_size)

#         embeds = self.embedding(batch_data)

#         if self.use_dropout:
#             embeds = self.dropout(embeds)

#         self.rnn.flatten_parameters()
#         embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

#         rnn_output, hidden = self.rnn(embeds, hidden)
#         if self.rnn_model.lower() == "lstm":
#             hidden = hidden[0]

#         rnn_output = pad_packed_sequence(rnn_output)[0]

#         rnn_output = rnn_output.permute(1, 0, 2)

#         weighted_outputs, attention_weights = perform_attention(self, rnn_output,
#                                                                 self.get_last_hidden_output(hidden)
#                                                                 )
#         return weighted_outputs, attention_weights

#     def get_last_hidden_output(self, hidden):
#         if self.bidirectional:
#             hidden_forward = hidden[-1]
#             hidden_backward = hidden[0]
#             if len(hidden_backward.shape) > 2:
#                 hidden_forward = hidden_forward.squeeze(0)
#                 hidden_backward = hidden_backward.squeeze(0)
#             last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
#         else:

#             last_rnn_output = hidden[-1]
#             if len(hidden.shape) > 2:
#                 last_rnn_output = last_rnn_output.squeeze(0)

#         return last_rnn_output

# -*- coding: utf-8 -*-
"""
    Word-based RNN model for text classification
    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 07/03/2019
    @date last modified: 19/08/2020
"""

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models.attentions.util import *
from src.models.embeddings.util import *
from src.data_helpers.vocab import Vocab
from src.models.multires_cnn import *
from src.models.multires_rnn import *
from src.models.gcn import *

adj_type   = 'occ_tboth'
emd_type   = 'dim_300_10_half_raw_50_w2v_avg_w2v_no_tfidf'

# simple cnn
# add output_size in args



# args
# self.args.cnn_filter_size
# self.args.cnn_att

# python3 -m src.run         --problem_name mimic-iii_2_50        --max_seq_length 4000         --n_epoch 50         --patience 5         --batch_size 8         --optimiser adamw         --lr 0.001         --dropout 0.3         --level_projection_size 128         --main_metric micro_f1         --exp_name experiment_base         --embedding_mode word2vec         --embedding_file data/embeddings/word2vec_sg0_100.model         --attention_mode label         --d_a 512         RNN          --rnn_model LSTM         --n_layers 1         --bidirectional 1         --hidden_size 512



class RNN(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):
        """
        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        self.output_size = self.hidden_size * self.n_directions
        self.rnn_model = args.rnn_model

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)

        weighted_outputs, attention_weights = perform_attention(self, rnn_output, self.get_last_hidden_output(hidden)
                                                                )
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output

    

    
    
    
# simple cnn
# add output_size in args



# args
# self.args.cnn_filter_size
# self.args.cnn_att



# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_CNN --cnn_filter_size 500

# self.args.cnn_att



class RNN_cnn(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_cnn, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.multires_cn = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)
        self.output_size = self.args.cnn_filter_size * 6
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        
        cnn_output = self.multires_cn(embeds)

        weighted_outputs, attention_weights = perform_attention(self, cnn_output)
        return weighted_outputs, attention_weights
    
 
#  self.args.gcn_drop
# self.args.gcn_att
# self.args.gcn_both

# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --attention_mode label --d_a 512 RNN_GCN --gcn_both 1

# self.args.gcn_att
# self.args.gcn_both




class RNN_gcn(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_gcn, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        
        
        
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)
        
        if self.args.gcn_both:
            self.output_size = self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = self.n_labels[1]
            
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        
        
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
        
        weighted_outputs, attention_weights = perform_attention(self, gcn_output)
        return weighted_outputs, attention_weights
    
    
# self.args.rnn_att

# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --attention_mode label --d_a 512 RNN_BIGRU --rnn_att 1
    
    
# self.args.rnn_att


    
class RNN_BIGRU(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_BIGRU, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        
        self.output_size  = 1024
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)
            
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        bigru_output = self.multires_rnn(embeds, lengths.cpu())
        weighted_outputs, attention_weights = perform_attention(self, bigru_output)
        return weighted_outputs, attention_weights
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --attention_mode label --d_a 512 RNN_CNN_CON --rnn_model LSTM --n_layers 1 --bidirectional 1 --hidden_size 512 --cnn_filter_size 500


# self.args.cnn_att


    
class RNN_CNN_CON(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):
        """
        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN_CNN_CON, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        self.output_size = (self.hidden_size * self.n_directions) + self.args.cnn_filter_size * 6
        self.rnn_model = args.rnn_model

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.multires_cn = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)
        
        cnn_output = self.multires_cn(embeds)
        
        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)
        
        df         = torch.cat((rnn_output, cnn_output), 2)        
        

        weighted_outputs, attention_weights = perform_attention(self, df ,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --attention_mode label --d_a 512 RNN_BIGRU_CON --rnn_model LSTM --n_layers 1 --bidirectional 1 --hidden_size 512 --rnn_att 1



# self.args.rnn_att


    
class RNN_BIGRU_CON(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):
        """
        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN_BIGRU_CON, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        self.output_size = (self.hidden_size * self.n_directions) + 1024
        self.rnn_model = args.rnn_model

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)
        
        bigru_output = self.multires_rnn(embeds, lengths.cpu())
        
        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)
        
        df         = torch.cat((rnn_output, bigru_output), 2)        
        
        weighted_outputs, attention_weights = perform_attention(self, df ,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
    
    
    
# self.args.gcn_att
# self.args.gcn_both


    
class RNN_GCN_CON(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):
        """
        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN_GCN_CON, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode

        self.rnn_model = args.rnn_model

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)
        
        if self.args.gcn_both:
            self.output_size = (self.hidden_size * self.n_directions) + self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = (self.hidden_size * self.n_directions) + self.n_labels[1]
            

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)
        
        
        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)
        
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
        
        df = torch.cat((rnn_output, gcn_output), 2)

        weighted_outputs, attention_weights = perform_attention(self, df ,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_cnn_bigru_con --cnn_filter_size 500 --rnn_att 0

# self.args.cnn_att
# self.args.rnn_att


    
class RNN_cnn_bigru_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_cnn_bigru_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode
        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.multires_cn = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)
        self.output_size = (self.args.cnn_filter_size * 6) + 1024
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        
        cnn_output = self.multires_cn(embeds)
        rnn_out    = self.multires_rnn(embeds, lengths.cpu())
        
        df         = torch.cat((cnn_output, rnn_out), 2)

        weighted_outputs, attention_weights = perform_attention(self, df)
        return weighted_outputs, attention_weights
    
#     python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_cnn_gcn_con --cnn_filter_size 500 --gcn_both 1


# self.args.cnn_att
# self.args.gcn_att
# self.args.gcn_both


    
class RNN_cnn_gcn_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_cnn_gcn_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode
        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.multires_cn = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)
        
        
        if self.args.gcn_both:
            self.output_size = (self.args.cnn_filter_size * 6) + self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = (self.args.cnn_filter_size * 6) + self.n_labels[1]
            
            
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)
        
        
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)
        
        if self.args.gcn_both:
            self.output_size = self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = self.n_labels[1]


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        
        cnn_output = self.multires_cn(embeds)
        
        
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
        
        
        
        df         = torch.cat((cnn_output, gcn_output), 2)

        weighted_outputs, attention_weights = perform_attention(self, df)
        return weighted_outputs, attention_weights
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --attention_mode label --d_a 512 RNN_BIGRU_GCN_CON --gcn_both 1
# self.args.gcn_att
# self.args.gcn_both
# self.args.rnn_att




class RNN_BIGRU_GCN_CON(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_BIGRU_GCN_CON, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        
        if self.args.gcn_both:
            self.output_size = 1024 + self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = 1024 + self.n_labels[1]

        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)
        
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)
            
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        bigru_output = self.multires_rnn(embeds, lengths.cpu())
        
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
            
        
        df         = torch.cat((bigru_output, gcn_output), 2)
        
        
        weighted_outputs, attention_weights = perform_attention(self, df)
        return weighted_outputs, attention_weights
    
    
    
    
    
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_rnn_cnn_bigru_con --cnn_filter_size 500 

# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_rnn_cnn_bigru_con --cnn_filter_size 500


# self.args.rnn_att
# self.args.cnn_att


    
class RNN_rnn_cnn_bigru_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_rnn_cnn_bigru_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        
        self.rnn_model = args.rnn_model
        self.embedding = init_embedding_layer(args, vocab)
        
        self.output_size = (self.hidden_size * self.n_directions) + (self.args.cnn_filter_size * 6) + 1024
        
        self.multires_cn = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)

        
        self.dropout = args.dropout

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)
            
            
        cnn_out   = self.multires_cn(embeds)
        bigru_out = self.multires_rnn(embeds, lengths.cpu())

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)
        
        df = torch.cat((rnn_output, cnn_out, bigru_out), 2)        

        weighted_outputs, attention_weights = perform_attention(self, df,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
    
    
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_rnn_cnn_gcn_con --gcn_both 0
    
# self.args.cnn_att
# self.args.gcn_both
# self.args.gcn_att


    
class RNN_rnn_cnn_gcn_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_rnn_cnn_gcn_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        
        self.rnn_model = args.rnn_model
        self.embedding = init_embedding_layer(args, vocab)
        
        if self.args.gcn_both:
            self.output_size = (self.hidden_size * self.n_directions) + (self.args.cnn_filter_size * 6)
            self.output_size = self.output_size + self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = (self.hidden_size * self.n_directions) + (self.args.cnn_filter_size * 6)
            self.output_size = self.output_size + self.n_labels[1]
        
        
        
        self.multires_cn = MultiResCNN(self.embedding.output_size, 
                                       self.args.cnn_filter_size, 
                                       cnn_att = self.args.cnn_att)

        
        self.dropout = args.dropout

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
            
            
            
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)
            
            
        cnn_out   = self.multires_cn(embeds)
        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)
        
        
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
        
        df = torch.cat((rnn_output, cnn_out, gcn_output), 2)
        weighted_outputs, attention_weights = perform_attention(self, df,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
    
    
# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_rnn_gcn_bigru_con --gcn_both 1


# self.args.rnn_att
# self.args.gcn_att
# self.args.gcn_both


    
class RNN_rnn_gcn_bigru_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_rnn_gcn_bigru_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        
        self.rnn_model = args.rnn_model
        self.embedding = init_embedding_layer(args, vocab)
                
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)
        
        if self.args.gcn_both:
            self.output_size = (self.hidden_size * self.n_directions) + 1024
            self.output_size = self.output_size + self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = (self.hidden_size * self.n_directions) + 1024
            self.output_size = self.output_size + self.n_labels[1]
            
            
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)

        
        self.dropout = args.dropout

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)
            
            
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
            
            
        bigru_out = self.multires_rnn(embeds, lengths.cpu())

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)
        
        df = torch.cat((rnn_output, gcn_output, bigru_out), 2)        

        weighted_outputs, attention_weights = perform_attention(self, df,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
    
    



# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_cnn_gcn_bigru_con --gcn_both 0 --cnn_filter_size 500
    
    
    
# self.args.cnn_att
# self.args.gcn_att
# self.args.gcn_both
# self.args.rnn_att


    
class RNN_cnn_gcn_bigru_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_cnn_gcn_bigru_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.attention_mode = args.attention_mode
        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.multires_cn  = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)
        
        if self.args.gcn_both:
            self.output_size = (self.args.cnn_filter_size * 6) + self.n_labels[0] + self.n_labels[1] + 1024
        else:
            self.output_size = (self.args.cnn_filter_size * 6) + self.n_labels[1] + 1024
            
            
        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)
        
        
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)
        
        if self.args.gcn_both:
            self.output_size = self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = self.n_labels[1]


    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        
        cnn_output = self.multires_cn(embeds)
        bigru_out  = self.multires_rnn(embeds, lengths.cpu())
        
        
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
        
                
        df         = torch.cat((cnn_output, gcn_output, bigru_out), 2)
        weighted_outputs, attention_weights = perform_attention(self, df)
        return weighted_outputs, attention_weights
    
    
    
    

# python3 -m src.run --problem_name mimic-iii_2_50 --max_seq_length 4000 --n_epoch 50 --patience 5 --batch_size 8 --optimiser adamw --lr 0.001 --dropout 0.3 --level_projection_size 128 --main_metric micro_f1 --exp_name experiment_base --embedding_mode word2vec --embedding_file data/embeddings/word2vec_sg0_100.model --output_size 3000 --attention_mode label --d_a 512 RNN_rnn_gcn_bigru__cnn_con --gcn_both 1 --cnn_filter_size 50

# self.args.cnn_att
# self.args.gcn_att
# self.args.gcn_both
# self.args.rnn_att




class RNN_rnn_gcn_bigru__cnn_con(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):

        super(RNN_rnn_gcn_bigru__cnn_con, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.n_labels = vocab.all_n_labels()
        self.n_level  = vocab.n_level()
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        
        self.rnn_model = args.rnn_model
        self.embedding = init_embedding_layer(args, vocab)
                
        self.multires_rnn = BiGRU(rnn_att = self.args.rnn_att, rnn_dim=1024)
        self.multires_cn  = MultiResCNN(self.embedding.output_size, self.args.cnn_filter_size, cnn_att = self.args.cnn_att)
        
        if self.args.gcn_both:
            self.output_size = (self.hidden_size * self.n_directions) + 1024 + (self.args.cnn_filter_size * 6)
            self.output_size = self.output_size + self.n_labels[0] + self.n_labels[1]
        else:
            self.output_size = (self.hidden_size * self.n_directions) + 1024 + (self.args.cnn_filter_size * 6)
            self.output_size = self.output_size + self.n_labels[1]
            
            
        self.gcn_1_data  = gc_data(self.n_labels, adj_type, self.args.gpu_id, emd_type)
        self.gcn_1_layer = gcn_l(self.n_labels,  self.gcn_1_data[0][0].shape[-1], 
                             self.args.max_seq_length, inner_dims = 1024, 
                             drop = self.args.gcn_drop, 
                             att  = self.args.gcn_att, device = self.args.gpu_id)

        
        self.dropout = args.dropout

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.args.gpu_id)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)
        
        

        if self.use_dropout:
            embeds = self.dropout(embeds)
            
        cnn_out = self.multires_cn(embeds)
        gcn_output0 = self.gcn_1_layer[0](self.gcn_1_data[0]).t()
        gcn_output1 = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
        if self.args.gcn_both:
            gcn_output = torch.cat((gcn_output0, gcn_output1), 1)
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])

        else:
            gcn_output = self.gcn_1_layer[1](self.gcn_1_data[1]).t()
            gcn_output = gcn_output.expand(batch_size, 
                                           gcn_output.size()[0], 
                                           gcn_output.size()[1])
            
            
        bigru_out = self.multires_rnn(embeds, lengths.cpu())

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)
        
        df = torch.cat((rnn_output, gcn_output, bigru_out, cnn_out), 2)
        weighted_outputs, attention_weights = perform_attention(self, df,
                                                                self.get_last_hidden_output(hidden))
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
