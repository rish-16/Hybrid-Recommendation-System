import numpy as np
import pandas as pd
import lightfm
import scipy as sp
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Import dataset
dataset = fetch_movielens(min_rating=4.0)

# Create model with loss function using Weighted Approximate-Rank Pairwise (warp)
model = LightFM(loss='warp')
model.fit(dataset['train'], epochs=30, num_threads=2)

# Generate recommendations
def sample_rec(model, data, user_ids):
	n_users, n_items = dataset['train'].shape

	for user_id in user_ids:
		known_pos = dataset['item_labels'][dataset['train'].tocsr()[user_id].indices]

		scores = model.predict(user_id, np.arange(n_items))

		top_items = dataset['item_labels'][np.argsort(-scores)]

		print ('_______________________________________________________________')
		print ('')

		print ('User {}'.format(user_id))
		print ('')
		print ('Known Positives:')

		for x in known_pos[:10]:
			print ('{}'.format(x))

		print ('')
		print ('')

		print ('Recommended movies:')

		for x in top_items[:5]:
			print ('{}'.format(x))

		print ('')
		print ('_______________________________________________________________')

# Provide list of various user IDs to obtain recommendations
sample_rec(model, dataset, [826])
