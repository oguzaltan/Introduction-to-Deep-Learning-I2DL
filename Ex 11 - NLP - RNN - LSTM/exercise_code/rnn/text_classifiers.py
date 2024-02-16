import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }

        self.hparams = hparams

        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        
        # Define the convolutional layers 
        self.embedding = Embedding(num_embeddings, embedding_dim, 0)
        self.rnn = RNN(embedding_dim, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lengths is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        embeddings = self.embedding(sequence)
        
        # Sequences need to have the same length
        # For this, we use the function pack_padded_sequence to truncate longer reviews 
        # and pad shorter reviews with zeros
        if lengths is not None:
            embeddings = pack_padded_sequence(embeddings, lengths)
        
        h_seq, h = self.rnn(embeddings)

        # print('h size:', h.size())
        h = h.squeeze(0)
        # print('Squeezed h size:', h.size())
        
        output = self.output(h)
        output = output.sigmoid().view(-1)
        
        # Outputs show probability of being positive for
        # each sequence (which is review) in the batch
        # If there are three sequences namely three reviews in the batch,
        # then output will have three numbers, one for each sequence in the batch
        # print('Output size', output.size())
        # print(output)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output
