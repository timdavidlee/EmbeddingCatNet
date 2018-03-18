import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F


def norm_init_emb(embedding):
    """
    Limit the random initialization 
    of embeddings to a specific cutoff
    1/dim, Doesn't return anything
    implicitly changes the data inplace
    """
    vec_dim = embedding.weight.data.size(1)
    sc = 2 / (vec_dim + 1)
    embedding.weight.data.uniform_(-sc, sc)


class EmbeddingModel(nn.Module):
    """
    Model at a high level, is made of of the following parts:

    1. Collection of Embeddings ( for all categorical vars)
    --------------------------------------------------------
        - each field will have a matrix associated with it
        - the row represents the values
        - the columns represent a embedding dimension (arbitrary)
        - generally don't want to choose a embedding dimension
          greater than the cardinality. If you have 2 unique choices
          for a field, don't make your vector dimension 50

    2. A shallow Neural network
    -----------------------------
        - will use linear layers in combination with batch normalization
          and some dropout to reduce overfitting

    3. A classification layer
    -----------------------------
        - for this particular problem, a classification layer was attached
          either a sigmoid or a log_softmax for predicting multiclass

    """
    def __init__(self, emb_szs, layer_sizes, output_dim, drop_pct, emb_drop_pct):

        """
        Initializes the embedding class
        layer_sizes: Expects a list [ 1000, 300, 100] means that 
                     3 linear layers will be made with 1000, 300, 100 
                     respectively
        emb_szs: [(5,2), (1000, 50), (10,5)] - each tuple represents
                 the level of cardinality (5)
                 and the desired level of embedding, (,2). Which 
                 we are using a rough rule of half cardinality, but
                 no greater than 50

        drop_pct : a float from 0 to 1, this value is how much dropout
                   to add to the sub-unit stack, consider this the level
                   of regularization. 1 is 100% drop out = no data

        emb_drop_pct: a float from 0 to 1, this is how much initial dropout
                      to apply to the 
        """
        super(EmbeddingModel, self).__init__()

        # Number of layers
        self.output_dim = output_dim
        self.n_layers = len(layer_sizes)
        print('number of feats:% d' % len(emb_szs))

        # initialize the embeddings
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_szs])
        self.total_emb_var_ct = 0
        for emb in self.embs:
            norm_init_emb(emb)
            self.total_emb_var_ct += emb.embedding_dim

        self.emb_drop = nn.Dropout(emb_drop_pct)
        print('total embedding parameters %d' % self.total_emb_var_ct)

        # initialize the layers, will make as many
        # sub blocks as passed in by layer_size list
        # ex. [n1, n2, n3] => 3 lin layer network + 1 output layer
        self.seq_model = nn.Sequential()

        # take all total embedding size and append it to the
        # front of the layer sizes
        layer_sizes = [self.total_emb_var_ct] + layer_sizes

        # Create all the blocks for the neural network
        # the sub-unit is:

        for i in range(self.n_layers):
            # One Subunit:
            # -- Linear
            # -- Relu
            # -- BatchNorm
            # -- Drop_out
            # gets the current layer size from 
            current_layer_size = layer_sizes[i]
            next_layer_size = layer_sizes[i + 1]

            # initialize the linear layer
            linlayer = nn.Linear(current_layer_size, next_layer_size)
            kaiming_normal(linlayer.weight.data)

            self.seq_model.add_module("lin_%d" % i, linlayer)
            self.seq_model.add_module("relu_%d" % i, nn.ReLU())
            self.seq_model.add_module("batch_norm_%d" % i, nn.BatchNorm1d(next_layer_size))
            self.seq_model.add_module("drop_out_%d" % i, nn.Dropout(drop_pct))

        out_lin = nn.Linear(layer_sizes[-1], output_dim)
        kaiming_normal(out_lin.weight.data)

        self.seq_model.add_module("output", out_lin)

        # initialize the weights of linear layers
        for ly in self.seq_model:
            if type(ly) == nn.Linear:
                kaiming_normal(ly.weight.data)

    def forward(self, x):
        # the regular data row comes in
        # for each value, lookup the corresponding vector
        lkups = [emb(x[:, idx]) for idx, emb in enumerate(self.embs)]

        # add all the vector dataframes together        
        glued = torch.cat(lkups, 1)        
        x = self.emb_drop(glued)

        # pass the unified data to the multi-layer
        # NN model
        x = self.seq_model(x)
        if self.output_dim == 1:
            # binary
            x = F.sigmoid(x)
        else:
            # multiclass classification
            x = F.log_softmax(x, self.output_dim)
        return x
