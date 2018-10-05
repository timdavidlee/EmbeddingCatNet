import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal


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
    def __init__(self,
                 emb_szs,
                 cat_cols=None,
                 cont_cols=None,
                 idx2col=None,
                 col2idx=None,
                 layer_sizes=[1000, 300],
                 output_dim=1,
                 drop_pct=0.2,
                 emb_drop_pct=0.2):

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

        # collect input variables
        self.emb_szs = emb_szs
        self.emb_cols = list(emb_szs.keys())
        self.idx2col = idx2col
        self.col2idx = col2idx
        self.output_dim = output_dim
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.cat_cols_ct = len(set(cat_cols).difference(set(self.emb_cols)))

        if cont_cols is None:
            self.cont_cols_ct = 0
        else:
            self.cont_cols_ct = len(cont_cols)
        # Number of layers
        self.n_layers = len(layer_sizes)

        # will initialize embeddings
        self.embs_nn = None
        self.total_emb_var_ct = 0
        self._init_embeddings(emb_szs)

        self.emb_drop = nn.Dropout(emb_drop_pct)
        print('total embedding parameters %d' % self.total_emb_var_ct)

        # initialize the layers, will make as many
        # sub blocks as passed in by layer_size list
        # ex. [n1, n2, n3] => 3 lin layer network + 1 output layer
        # take all total embedding size and append it to the
        # front of the layer sizes
        layer_sizes = [self.total_emb_var_ct + self.cont_cols_ct + self.cat_cols_ct] + layer_sizes

        self.seq_model = self._init_layers(layer_sizes, output_dim, drop_pct)

    def _init_layers(self, layer_sizes, output_dim, drop_pct):
        """
        Initialize the layers of the embedding model
        """
        n_layers = len(layer_sizes)
        seq_model = nn.Sequential()
        # Create all the blocks for the neural network
        # the sub-unit is:
        for i in range(n_layers - 1):
            # gets the current layer size from
            current_layer_size = layer_sizes[i]
            next_layer_size = layer_sizes[i + 1]

            # initialize the linear layer
            linlayer = nn.Linear(current_layer_size, next_layer_size)
            kaiming_normal(linlayer.weight.data)

            # One Subunit:
            # -- Linear
            # -- Relu
            # -- BatchNorm
            # -- Drop_out
            seq_model.add_module("lin_%d" % i, linlayer)
            seq_model.add_module("relu_%d" % i, nn.ReLU())
            seq_model.add_module("batch_norm_%d" % i,
                                 nn.BatchNorm1d(next_layer_size))
            seq_model.add_module("drop_out_%d" % i, nn.Dropout(drop_pct))

        out_lin = nn.Linear(layer_sizes[-1], output_dim)
        kaiming_normal(out_lin.weight.data)

        seq_model.add_module("output", out_lin)

        # initialize the weights of linear layers
        for ly in seq_model:
            if type(ly) == nn.Linear:
                kaiming_normal(ly.weight.data)

        return seq_model

    def _init_embeddings(self, emb_szs):
        print('number of emb feats:% d' % len(emb_szs.keys()))
        # initialize the embeddings
        self.embs_nn = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_szs.values()])
        for emb in self.embs_nn:
            norm_init_emb(emb)
            self.total_emb_var_ct += emb.embedding_dim

    def forward(self, x):
        # the regular data row comes in
        # for each value, lookup the corresponding vector
        lkups = []
        x_indices = set([v for v in range(x.shape[1])])
        emb_indices = []

        for idx, emb in enumerate(self.embs_nn):
            colname = self.emb_cols[idx]
            colidx = self.col2idx[colname]
            emb_indices.append(colidx)
            lkups.append(emb(x[:, colidx]))

        emb_indices = set(emb_indices)
        non_emb_indices = x_indices.difference(emb_indices)

        # add all the vector dataframes together
        glued = torch.cat(lkups, 1)
        xx = self.emb_drop(glued)

        if len(non_emb_indices) > 0:
            xx = torch.cat([xx, x[:, list(non_emb_indices)].float()], 1)

        # pass the unified data to the multi-layer
        # NN model
        xx = self.seq_model(xx)
        if self.output_dim == 1:
            # binary
            xx = F.sigmoid(xx)
        else:
            # multiclass classification
            xx = F.log_softmax(xx, 0)
        return xx


def train_model(model, train_loader, loss_fn, **kwargs):
    n_epochs = kwargs['n_epoches']
    wd = kwargs['weight_decay']
    lr = kwargs['learning_rate']
    ml_type = kwargs['ml_type']
    if ml_type == 'multi':
        n_classes = kwargs['n_classes']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    avg_loss = []
    for learning_rate in [0.01, 0.003, 0.001, 0.0003, 0.0001]:
        print('learning rate %f' % learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_correct = 0

            train_dl = iter(train_loader)
            running_loss = 0.0
            for i, batch in enumerate(train_dl):
                data, labels = batch
                bz = data.size()[0]

                data_var = Variable(data)
                label_var = Variable(labels.float(), requires_grad=False)

                y_pred = model(data_var)

                if ml_type == 'binary':
                    y_pred_hard = y_pred > 0.5
                    correct = (label_var.view(-1, 1).eq(y_pred_hard.float())).sum()
                    loss = loss_fn(y_pred, label_var)
                elif ml_type == 'multi':
                    label_num = torch.max(label_var, 1)[1]
                    pred_num = torch.max(y_pred, 1)[1]
                    correct = label_num.eq(pred_num).sum()
                    loss = loss_fn(y_pred, label_var)

                running_correct += correct.float().data

                running_loss += loss.data[0]

                if i % 25 == 24:
                    avg_loss.append(running_loss / 25)
                    acc = running_correct / 50 / 25.
                    print('[%d/%d] - %d/%d loss: %f, acc: %f' % (epoch + 1, n_epochs, i * bz, 18200, running_loss / 25, acc))

                    running_loss = 0
                    running_correct = 0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
