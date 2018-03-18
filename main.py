from utils import load_wids_xy_data, get_mappers
from EmbeddingModel import EmbeddingModel


def train_model(model, train_loader, n_epochs, optimizer, loss_fn)
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


if __name__ == '__main__':
	"""
	Load the wids kaggle data, the swap all the categoical data
	and create mappers
	"""
	X, y = load_wids_xy_data()
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

	# number of epoches to train
	n_epochs = 4
	weight_decay = 1e-5

	# select an optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

	# choose a loss function
	# since we are training binary, we choose a BCELoss
	# or binary cross entropy
	loss_fn = torch.nn.BCELoss(size_average=False)

	# initialize model
	model = EmbeddingModel(emb_szs=emb_szs, layer_sizes=layer_sizes, output_dim=1, drop_pct=0.3, emb_drop_pct=0.3)

	# show the network architecture to console
	print(model.seq_model)

	# train the model
	train_model(model, train_loader, n_epochs, optimizer, loss_fn)

