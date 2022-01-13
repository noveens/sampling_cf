from data import rating_data
from utils import remap_items
from data_path_constants import get_data_path

def prep(dataset):
	f = open(get_data_path(dataset) + '/data.csv', "r")
	users, items, ratings, time = [], [], [], []

	user_map, item_map = {}, {}

	line = f.readline()
	temp = line.strip().split(",")
	col_map = dict(zip(temp, range(len(temp))))

	line = f.readline() # First line was header
	while line:
		datum = line.strip().split(",") ; line = f.readline()
		if len(datum) != len(col_map): continue

		u = datum[col_map['review_profilename']]
		i = datum[col_map['beer_beerid']]
		r = datum[col_map['review_overall']]
		t = datum[col_map['review_time']]

		if u not in user_map: user_map[u] = len(user_map)
		if i not in item_map: item_map[i] = len(item_map)

		users.append(user_map[u])
		items.append(item_map[i])
		ratings.append(float(r))
		time.append(int(t))

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
