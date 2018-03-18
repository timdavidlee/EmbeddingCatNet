from utils import load_wids_xy_data, get_mappers
from EmbeddingModel import EmbeddingModel


def train_model(n_epochs, weight_decay, optimizer, loss_fn)

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
	            y_pred_hard = y_pred > 0.5
	            correct = (label_var.view(-1,1).eq(y_pred_hard.float())).sum()
	            running_correct += correct.float().data    

	            loss = loss_fn(y_pred, label_var)
	            running_loss += loss.data[0]

	            if i % 25 == 24:
	                avg_loss.append(running_loss/25)
	                acc = running_correct/50/25.
	                print('[%d/%d] - %d/%d loss: %f, acc: %f' %(epoch+1, n_epochs, i*bz, 18200, running_loss/25, acc))
	                
	                running_loss = 0
	                running_correct = 0

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

	layer_sizes = [1000, 300, 100]
	n_epochs = 4
	weight_decay = 1e-5
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
	loss_fn = torch.nn.BCELoss(size_average=False)

	# initialize 
	model = EmbeddingModel(emb_szs=emb_szs, layer_sizes=layer_sizes, output_dim=1, drop_pct=0.3, emb_drop_pct=0.3)
	model.seq_model

