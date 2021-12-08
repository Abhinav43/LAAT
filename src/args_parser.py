# import argparse
# from src.models.rnn import *
# from src.models.cnn import *


# def create_args_parser():
#     parser = argparse.ArgumentParser(description="DNN for Text Classifications")
#     parser.add_argument("--problem_name", type=str, default="mimic-iii_single_full", required=False,
#                         help="The problem name is used to load the configuration from config.json")

#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument("--n_epoch", type=int, default=50)
#     parser.add_argument("--patience", type=int, default=5, help="Early Stopping")

#     parser.add_argument("--optimiser", type=str, choices=["adagrad", "adam", "sgd", "adadelta", "adamw"],
#                         default="adamw")
#     parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
#     parser.add_argument("--weight_decay", type=float, default=0)

#     parser.add_argument("--use_lr_scheduler", type=int, choices=[0, 1], default=1,
#                         help="Use lr scheduler to reduce the learning rate during training")
#     parser.add_argument("--lr_scheduler_factor", type=float, default=0.9,
#                         help="Reduce the learning rate by the scheduler factor")
#     parser.add_argument("--lr_scheduler_patience", type=int, default=5,
#                         help="The lr scheduler patience")

#     parser.add_argument("--joint_mode", type=str, choices=["flat", "hierarchical"], default="hierarchical")
#     parser.add_argument("--level_projection_size", type=int, default=128)

#     parser.add_argument("--main_metric", default="micro_f1",
#                         help="the metric to be used for validation",
#                         choices=["macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
#                                  "micro_accuracy", "micro_precision", "micro_recall", "micro_f1", "micro_auc", "loss",
#                                  "macro_P@1", "macro_P@5", "macro_P@8", "macro_P@10", "macro_P@15"])
#     parser.add_argument("--metric_level", type=int, default=1,
#                         help="The label level to be used for validation:"
#                              "\n\tn: The n-th level if n >= 0 (started with 0)"
#                              "\n\tif n > max_level, n is set to max_level"
#                              "\n\tif n < 0, use the average of all label levels"
#                              )

#     parser.add_argument("--multilabel", default=1, type=int, choices=[0, 1])

#     parser.add_argument("--shuffle_data", type=int, choices=[0, 1], default=1)

#     parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")

#     parser.add_argument("--save_best_model", type=int, choices=[0, 1], default=1)
#     parser.add_argument("--save_results", type=int, choices=[0, 1], default=1)
#     parser.add_argument("--best_model_path", type=str, default=None)
#     parser.add_argument("--save_results_on_train", action='store_true', default=False)
#     parser.add_argument("--resume_training", action='store_true', default=False)

#     parser.add_argument("--max_seq_length", type=int, default=4000)
#     parser.add_argument("--min_seq_length", type=int, default=-1)
#     parser.add_argument("--min_word_frequency", type=int, default=-1)

#     # Embedding
#     parser.add_argument("--mode", type=str, default="static",
#                         choices=["rand", "static", "non_static", "multichannel"],
#                         help="The mode to init embeddings:"
#                              "\n\t1. rand: initialise the embedding randomly"
#                              "\n\t2. static: using pretrained embeddings"
#                              "\n\t2. non_static: using pretrained embeddings with fine tuning"
#                              "\n\t2. multichannel: using both static and non-static modes")

#     parser.add_argument("--embedding_mode", type=str, default="fasttext",
#                         help="Choose the embedding mode which can be fasttext, word2vec")
#     parser.add_argument('--embedding_size', type=int, default=100)
#     parser.add_argument("--embedding_file", type=str, default=None)
#     parser.add_argument("--exp_name", type=str, default=None)
#     parser.add_argument("--loss_name", type=str, default='base')
#     parser.add_argument("--reduction", type=str, default='mean')
    
    
    

#     # Attention
#     parser.add_argument("--attention_mode", type=str, choices=["hard", "self", "label", "caml"], default=None)
#     parser.add_argument("--d_a", type=int, help="The dimension of the first dense layer for self attention", default=-1)
#     parser.add_argument("--r", type=int, help="The number of hops for self attention", default=-1)
#     parser.add_argument("--use_regularisation", action='store_true', default=False)
#     parser.add_argument("--penalisation_coeff", type=float, default=0.01)

#     sub_parsers = parser.add_subparsers()

#     _add_sub_parser_for_cnn(sub_parsers)
#     _add_sub_parser_for_rnn(sub_parsers)

#     return parser


# def _add_sub_parser_for_rnn(subparsers):
#     args = subparsers.add_parser("RNN")
#     args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
#     args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
#                       help="Whether or not using bidirectional connection")
#     args.set_defaults(model=RNN)

#     args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
#                       help="Using the last hidden state or using the average of all hidden state")

#     args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")


# def _add_sub_parser_for_cnn(subparsers):
#     args = subparsers.add_parser("CNN")
#     args.add_argument("--cnn_model", type=str, choices=["CONV1D", "TCN"], default="CONV1D")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers for TCN")
#     args.add_argument("--out_channels", type=int, default=100, help="The number of out channels")
#     args.add_argument("--kernel_size", type=int, default=5, help="The kernel sizes.")
#     args.set_defaults(model=WordCNN)

import argparse
from src.models.rnn import *
from src.models.cnn import *


def create_args_parser():
    parser = argparse.ArgumentParser(description="DNN for Text Classifications")
    parser.add_argument("--problem_name", type=str, default="mimic-iii_single_full", required=False,
                        help="The problem name is used to load the configuration from config.json")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping")

    parser.add_argument("--optimiser", type=str, choices=["adagrad", "adam", "sgd", "adadelta", "adamw"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--use_lr_scheduler", type=int, choices=[0, 1], default=1,
                        help="Use lr scheduler to reduce the learning rate during training")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.9,
                        help="Reduce the learning rate by the scheduler factor")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5,
                        help="The lr scheduler patience")

    parser.add_argument("--joint_mode", type=str, choices=["flat", "hierarchical"], default="hierarchical")
    parser.add_argument("--level_projection_size", type=int, default=128)

    parser.add_argument("--main_metric", default="micro_f1",
                        help="the metric to be used for validation",
                        choices=["macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
                                 "micro_accuracy", "micro_precision", "micro_recall", "micro_f1", "micro_auc", "loss",
                                 "macro_P@1", "macro_P@5", "macro_P@8", "macro_P@10", "macro_P@15"])
    parser.add_argument("--metric_level", type=int, default=1,
                        help="The label level to be used for validation:"
                             "\n\tn: The n-th level if n >= 0 (started with 0)"
                             "\n\tif n > max_level, n is set to max_level"
                             "\n\tif n < 0, use the average of all label levels"
                             )

    parser.add_argument("--multilabel", default=1, type=int, choices=[0, 1])

    parser.add_argument("--shuffle_data", type=int, choices=[0, 1], default=1)

    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")

    parser.add_argument("--save_best_model", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_results", type=int, choices=[0, 1], default=1)
    parser.add_argument("--best_model_path", type=str, default=None)
    parser.add_argument("--save_results_on_train", action='store_true', default=False)
    parser.add_argument("--resume_training", action='store_true', default=False)

    parser.add_argument("--max_seq_length", type=int, default=4000)
    parser.add_argument("--min_seq_length", type=int, default=-1)
    parser.add_argument("--min_word_frequency", type=int, default=-1)

    # Embedding
    parser.add_argument("--mode", type=str, default="static",
                        choices=["rand", "static", "non_static", "multichannel"],
                        help="The mode to init embeddings:"
                             "\n\t1. rand: initialise the embedding randomly"
                             "\n\t2. static: using pretrained embeddings"
                             "\n\t2. non_static: using pretrained embeddings with fine tuning"
                             "\n\t2. multichannel: using both static and non-static modes")

    parser.add_argument("--embedding_mode", type=str, default="fasttext",
                        help="Choose the embedding mode which can be fasttext, word2vec")
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument("--embedding_file", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--attention_kp", type=str, default='base')
    parser.add_argument("--reduction", type=str, default='mean')
    parser.add_argument("--output_size", type=int, default=300)
    
    parser.add_argument("--gcn_drop", type=int, default=0)
#     parser.add_argument("--gcn_att", type=int, default=1)
#     parser.add_argument("--concat_1_att", type=int, default=1)
#     parser.add_argument("--concat_2_att", type=int, default=1)
#     parser.add_argument("--concat_3_att", type=int, default=1)
#     parser.add_argument("--final_att", type=int, default=1)

    
    # bottom layer args
    
#     parser.add_argument("--after_sum", type=str, default='catsum')
#     parser.add_argument("--after_sum", type=int, default=1)
    
    
    

    # Attention
    parser.add_argument("--attention_mode", type=str, choices=["hard", "self", "label", "caml"], default=None)
    parser.add_argument("--d_a", type=int, help="The dimension of the first dense layer for self attention", default=-1)
    parser.add_argument("--r", type=int, help="The number of hops for self attention", default=-1)
    parser.add_argument("--use_regularisation", action='store_true', default=False)
    parser.add_argument("--penalisation_coeff", type=float, default=0.01)

    sub_parsers = parser.add_subparsers()

    _add_sub_parser_for_cnn(sub_parsers)
    _add_sub_parser_for_rnn(sub_parsers)
    _add_sub_parser_for_rnn_cnn(sub_parsers)
    _add_sub_parser_for_rnn_gcn(sub_parsers)
    _add_sub_parser_for_rnn_bigru(sub_parsers)
    _add_sub_parser_for_rnn_cnn_con(sub_parsers)
    _add_sub_parser_for_rnn_bigru_con(sub_parsers)
    _add_sub_parser_for_rnn_gcn_con(sub_parsers)
    _add_sub_parser_for_rnn_cnn_bigru_con(sub_parsers)
    _add_sub_parser_for_rnn_cnn_gcn_con(sub_parsers)
    _add_sub_parser_for_rnn_bigru_gcn_con(sub_parsers)
    _add_sub_parser_for_RNN_rnn_cnn_bigru_con(sub_parsers)
    _add_sub_parser_for_RNN_rnn_cnn_gcn_con(sub_parsers)
    _add_sub_parser_for_RNN_rnn_gcn_bigru_con(sub_parsers)
    _add_sub_parser_for_RNN_cnn_gcn_bigru_con(sub_parsers)
    _add_sub_parser_for_RNN_rnn_gcn_bigru__cnn_con(sub_parsers)

    return parser


def _add_sub_parser_for_rnn(subparsers):
    args = subparsers.add_parser("RNN")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.set_defaults(model=RNN)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    

    
def _add_sub_parser_for_rnn_cnn_con(subparsers):
    args = subparsers.add_parser("RNN_CNN_CON")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_CNN_CON)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
def _add_sub_parser_for_rnn_bigru_con(subparsers):
    args = subparsers.add_parser("RNN_BIGRU_CON")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")

    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_BIGRU_CON)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    

def _add_sub_parser_for_rnn_gcn_con(subparsers):
    args = subparsers.add_parser("RNN_GCN_CON")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")

    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_GCN_CON)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
def _add_sub_parser_for_rnn_cnn(subparsers):
    args = subparsers.add_parser("RNN_CNN")
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.set_defaults(model=RNN_cnn)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")
    
    
def _add_sub_parser_for_rnn_cnn_bigru_con(subparsers):
    args = subparsers.add_parser("RNN_cnn_bigru_con")
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.set_defaults(model=RNN_cnn_bigru_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")
    
def _add_sub_parser_for_rnn_cnn_gcn_con(subparsers):
    args = subparsers.add_parser("RNN_cnn_gcn_con")
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.set_defaults(model=RNN_cnn_gcn_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")
    
    
    
    
    
    
    
def _add_sub_parser_for_rnn_bigru_gcn_con(subparsers):
    args = subparsers.add_parser("RNN_BIGRU_GCN_CON")
    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.set_defaults(model=RNN_BIGRU_GCN_CON)
    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")
    
def _add_sub_parser_for_RNN_rnn_cnn_bigru_con(subparsers):
    args = subparsers.add_parser("RNN_rnn_cnn_bigru_con")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    
    args.set_defaults(model=RNN_rnn_cnn_bigru_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
def _add_sub_parser_for_RNN_rnn_cnn_gcn_con(subparsers):
    args = subparsers.add_parser("RNN_rnn_cnn_gcn_con")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    
    args.set_defaults(model=RNN_rnn_cnn_gcn_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
def _add_sub_parser_for_RNN_rnn_gcn_bigru_con(subparsers):
    args = subparsers.add_parser("RNN_rnn_gcn_bigru_con")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")

    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    
    args.set_defaults(model=RNN_rnn_gcn_bigru_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
def _add_sub_parser_for_RNN_cnn_gcn_bigru_con(subparsers):
    args = subparsers.add_parser("RNN_cnn_gcn_bigru_con")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")

    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_cnn_gcn_bigru_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
def _add_sub_parser_for_RNN_rnn_gcn_bigru__cnn_con(subparsers):
    args = subparsers.add_parser("RNN_rnn_gcn_bigru__cnn_con")
    args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")

    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_drop", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.add_argument("--cnn_filter_size", type=int, default=50, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--cnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_rnn_gcn_bigru__cnn_con)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

    args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")
    
    
    
    
    RNN_rnn_gcn_bigru__cnn_con
    
    

    
#      self.args.gcn_drop
# self.args.gcn_att
# self.args.gcn_both
    
def _add_sub_parser_for_rnn_gcn(subparsers):
    args = subparsers.add_parser("RNN_GCN")
    args.add_argument("--gcn_drop", type=int, default=0, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
    args.add_argument("--gcn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    args.add_argument("--gcn_both", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_gcn)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")
    
    
def _add_sub_parser_for_rnn_bigru(subparsers):
    args = subparsers.add_parser("RNN_BIGRU")
    args.add_argument("--rnn_att", type=int, choices=[0, 1], default=1,
                      help="Whether or not using bidirectional connection")
    
    args.set_defaults(model=RNN_BIGRU)

    args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
                      help="Using the last hidden state or using the average of all hidden state")

def _add_sub_parser_for_cnn(subparsers):
    args = subparsers.add_parser("CNN")
    args.add_argument("--cnn_model", type=str, choices=["CONV1D", "TCN"], default="CONV1D")
    args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers for TCN")
    args.add_argument("--out_channels", type=int, default=100, help="The number of out channels")
    args.add_argument("--kernel_size", type=int, default=5, help="The kernel sizes.")
    args.set_defaults(model=WordCNN)
