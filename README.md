# Embeddings, Categorical Data, and Neural Networks


```
$ python main.py

	loading data ...
	complete ...
	formatting ...
	imputing missing values ...
	(18255, 1127) (18255,)
	converting to category ...
	calculating cardinality
	remapping columns to int
	complete
	
	# network architecture
	number of feats: 1127
	total embedding parameters 3636
	Sequential(
	  (lin_0): Linear(in_features=3636, out_features=1000)
	  (relu_0): ReLU()
	  (batch_norm_0): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True)
	  (drop_out_0): Dropout(p=0.3)
	  (lin_1): Linear(in_features=1000, out_features=300)
	  (relu_1): ReLU()
	  (batch_norm_1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True)
	  (drop_out_1): Dropout(p=0.3)
	  (lin_2): Linear(in_features=300, out_features=100)
	  (relu_2): ReLU()
	  (batch_norm_2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True)
	  (drop_out_2): Dropout(p=0.3)
	  (output): Linear(in_features=100, out_features=1)
	)
	
	# training output
	learning rate 0.010000
		[1/4] - 1200/18200 loss: 25.193178, acc: 0.753600
		[1/4] - 2450/18200 loss: 17.644042, acc: 0.837600
		[1/4] - 3700/18200 loss: 15.994287, acc: 0.858400
		[1/4] - 4950/18200 loss: 15.903909, acc: 0.864000
		[1/4] - 6200/18200 loss: 16.753692, acc: 0.856000
		[1/4] - 7450/18200 loss: 16.189508, acc: 0.839200
		[1/4] - 8700/18200 loss: 15.112801, acc: 0.880800
		[1/4] - 9950/18200 loss: 13.790286, acc: 0.892000
		...
	

```