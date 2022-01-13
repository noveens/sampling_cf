import json
from tqdm import tqdm
from datetime import datetime, timezone

from data import rating_data
from utils import remap_items
from data_path_constants import get_data_path

def prep(dataset):
	num_lines = sum(1 for line in open(get_data_path(dataset) + '/data.json', "r"))

	f = open(get_data_path(dataset) + '/data.json', "r")
	users, items, ratings, time = [], [], [], []
	utc_time = datetime.fromtimestamp(0, timezone.utc)

	user_map, item_map = {}, {}

	bar = tqdm(total = num_lines)
	line = f.readline()
	while line:
		bar.update(1)

		datum = json.loads(line.strip()) ; line = f.readline()
		u = datum['user_id']
		i = datum['book_id']
		r = datum['rating']
		if r == 0: continue
		
		t = datetime.strptime(datum['date_added'], '%a %b %d %H:%M:%S %z %Y')
		t = (t - utc_time).total_seconds()

		if u not in user_map: user_map[u] = len(user_map)
		if i not in item_map: item_map[i] = len(item_map)

		users.append(user_map[u])
		items.append(item_map[i])
		ratings.append(float(r))
		time.append(int(t))

	bar.close()
	f.close()

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
