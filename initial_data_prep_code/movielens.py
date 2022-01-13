from data import rating_data
from utils import remap_items
from data_path_constants import get_data_path

def prep(dataset):
	if dataset == "ml-100k": later_path, delim = "/u.data", "\t"
	elif dataset == "ml-25m": later_path, delim = "/ratings.csv", ","
	
	f = open(get_data_path(dataset) + later_path, "r")
	users, items, ratings, time = [], [], [], []

	line = f.readline()
	if dataset == "ml-25m": line = f.readline() # ml-25m has a header
	while line:
		u, i, r, t = line.strip().split(delim)
		users.append(int(u))
		items.append(int(i))
		ratings.append(float(r))
		time.append(int(t))
		line = f.readline()

	min_user = min(users) ; max_user = max(users)
	num_users = len(set(users))

	if min_user == 1:
		assert num_users == max_user
	else:
		assert num_users == max_user + 1

	data = [ [] for _ in range(num_users) ]
	for i in range(len(users)):
		data[users[i] - min_user].append([ items[i], ratings[i], time[i] ])

	# Time sort data
	for i in range(len(data)): 
		data[i].sort(key = lambda x: x[2]) 

	# Shuffling users
	# indices = np.arange(len(data)) ; np.random.shuffle(indices)
	# data = np.array(data)[indices].tolist()

	return rating_data(remap_items(data))
