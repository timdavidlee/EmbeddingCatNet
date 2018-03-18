from EmbeddingModel import EmbeddingModel
from utils import load_wids_xy_data, get_mappers

import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.modules.loss import BCELoss




def train_model(model, train_loader, n_epochs, optimizer, loss_fn):
    """
    Function used to train the embedding model
    n_epochs: number of epoches to train
    optimizer: pytorch optimizer
    loss_fn: usually BCELoss or MCELoss for classification
    """

    # for collecting statistics for charting later
    avg_loss = []

    # to make net converge smoother, the learning rate is changed
    # training will continue. Each learning rate with train for
    # n_epochs. So in the example below, 4 x 4 = 16 epoches total
    # similiar to an adaptive learning rate

    for learning_rate in [0.01, 0.003, 0.001, 0.0003, 0.0001]:

        # setting the new learning rate
        print('learning rate %f' % learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            
        for epoch in range(n_epochs):

            # collecting stats for console printing
            running_loss = 0.0
            running_correct = 0
            
            # create the dataloader
            train_dl = iter(train_loader)

            # start mini-batch training
            for i, batch in enumerate(train_dl):
                # unpack data and labels
                data, labels = batch                
                data_var = Variable(data)
                label_var = Variable(labels.float(), requires_grad=False)

                # infer batch size
                bz = data.size()[0]

                # predict with model
                y_pred = model(data_var)

                # calculate loss
                loss = loss_fn(y_pred, label_var)                

                # calculate hard predictions for running statistic (print to console)
                y_pred_hard = y_pred > 0.5
                correct = (label_var.view(-1,1).eq(y_pred_hard.float())).sum()
                running_correct += correct.float().data    

                # aggregate loss for running statistic (print to console)
                running_loss += loss.data[0]

                # status of training: print to console
                if i % 25 == 24:
                    avg_loss.append(running_loss/25)
                    acc = running_correct/50/25.
                    print('[%d/%d] - %d/%d loss: %f, acc: %f' %(epoch+1, n_epochs, i*bz, 18200, running_loss/25, acc))
                    
                    # reset the overview statistic
                    running_loss = 0
                    running_correct = 0

                # back propogation
                optimizer.zero_grad()
                loss.backward()        
                optimizer.step()


def get_embeddings_from_model(model):
    """
    Expects a EmbeddingModel which has the embeddings stored in

    - model.embs, which is a collection of torch.nn.Embedding objects

    The weights can be accessed via: model.embs[0].weight.data
    """    
    keys = mappers.keys()
    emb_mtx = {}
    for field, emb in zip(keys, model.embs):
        emb_mtx[field] = emb.weight.data.numpy()
    return emb_mtx


def get_emb_df(X, mappers, emb_mtx):
	"""
	Takes in :
	1. X: original data
	2. mappers: mapping categories to integers
	3. the trained vectors for the categories

	returns a new dataframe with all categories
	replaced with their vector formats
	"""


    cat_field_emb_dfs = []
    
    print('applying embeddings')
    for col in X.columns.values:
    	# for each field , remap categorical values --> ints
        idxs = X[col].map(mappers[col])
        
        # (for test data) fill new values /nones with new value 
        # (will be global mean)
        idxs[idxs.isna()] = max(idxs)+1
        idxs = np.array(idxs, dtype = int)
        
        # get embedding matrix for this field
        mtx = emb_mtx[col]
        
        # calculate global mean for missing values
        glb_mean = np.mean(mtx,axis=0)        
        
        # add global mean to bottom of matrix
        mtx = np.concatenate([mtx, glb_mean.reshape(1,-1)], axis=0)
        
        # use the categorical values --> ints as indices
        # to rearrange the embedding matrix by swapping rows
        # mtx[[2,2,2], :] - > will create a matrix with row 2 x 3 times
        jf = pd.DataFrame(mtx[idxs, :])
        jf.columns = [col+'_%d' %i for i in jf.columns]
        
        # collect the dataframe
        cat_field_emb_dfs.append(jf)

    # combine all the column vectors into one large dataframe
    # and return. Note, this will be much larger than the 
    # original dataframe
    print('combining dfs')
    out_df = pd.concat(cat_field_emb_dfs, axis=1)
    
    return out_df

if __name__ == '__main__':
    """
    Load the wids kaggle data, the swap all the categoical data
    and create mappers
    """
    X, y = load_wids_xy_data('~/data/wids/train.csv')
    X_mapped, mappers, categorical_stats, emb_szs = get_mappers(X)

    X_tensor = torch.from_numpy(X_mapped.head(18200).as_matrix())
    y_tensor = torch.from_numpy(y[:18200]).view(-1,1)

    # select a batch size and create dataloaders
    bz = 50
    train = data_utils.TensorDataset(X_tensor, y_tensor)
    train_loader = data_utils.DataLoader(train, batch_size=bz, shuffle=True)

    # model details
    # choose the linear layer network
    # the below sample will create 3 layers 1000, 300, 100
    # and one more additional layer as the fully connected output layer
    layer_sizes = [1000, 300, 100]

    # initialize model
    model = EmbeddingModel(emb_szs=emb_szs, layer_sizes=layer_sizes, output_dim=1, drop_pct=0.3, emb_drop_pct=0.3)

    # number of epoches to train
    n_epochs = 4
    weight_decay = 1e-5

    # select an optimizer
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # choose a loss function
    # since we are training binary, we choose a BCELoss
    # or binary cross entropy
    loss_fn = torch.nn.BCELoss(size_average=False)

    # show the network architecture to console
    print(model.seq_model)

    # train the model
    train_model(model, train_loader, n_epochs, optimizer, loss_fn)

    # now that the model is trained, we have embeddings that we can use for other models:
    emb_mtx = get_embeddings_from_model(model)

    # create an embedding-based dataframe (all categories swapped 
    # out for their embedding vectors)
    X_emb_df = get_emb_df(X, mappers, emb_mtx)

    """
	Can run XGBoost or any other model after this point
	simply have your 

	model = XGClassifier(...)
	model.fit(X_emb_df, y)

    """



