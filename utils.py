INF = float(1e6)

def get_data_loader_class(hyper_params):
    from data_loaders import MF, MVAE, SASRec, SVAE

    return {
        "pop_rec": (MF.TrainDataset, MF.TestDataset),
        "bias_only": (MF.TrainDataset, MF.TestDataset),
        "MF_dot": (MF.TrainDataset, MF.TestDataset),
        "MF": (MF.TrainDataset, MF.TestDataset),
        "NeuMF": (MF.TrainDataset, MF.TestDataset),
        "MVAE": (MVAE.TrainDataset, MVAE.TestDataset),
        "SVAE": (SVAE.TrainDataset, SVAE.TestDataset),
        "SASRec": (SASRec.TrainDataset, SASRec.TestDataset),
    }[hyper_params['model_type']]

def valid_hyper_params(hyper_params):
    ## Check if the methods and task match
    valid_tasks = {
        "pop_rec":      [             'implicit', 'sequential' ],
        "bias_only":    [ 'explicit', 'implicit', 'sequential' ],
        "MF_dot":       [ 'explicit', 'implicit', 'sequential' ],
        "MF":           [ 'explicit', 'implicit', 'sequential' ],
        "NeuMF":        [ 'explicit', 'implicit', 'sequential' ],
        "MVAE":         [             'implicit', 'sequential' ],
        "SVAE":         [                         'sequential' ],
        "SASRec":       [                         'sequential' ],
    }[hyper_params['model_type']]

    return hyper_params['task'] in valid_tasks

def get_common_path(hyper_params, star_match = False):
    ## E.g. Running SASRec on an explicit/implicit feedback task.
    if not valid_hyper_params(hyper_params): return None

    # To avoid writing hyper_params[key] everytime
    def get(key): 
        if star_match: return hyper_params.get(key, ".*")
        return hyper_params[key]

    common_path = "{}_{}".format(get('dataset'), get('task'))

    if get('sampling')[:3] == 'svp':
        common_path += "_{}_{}_perc_{}".format(get('sampling'), get('sampling_percent'), get('sampling_svp'))
    elif get('sampling') == 'complete_data': common_path += "_complete_data"
    else: common_path += "_{}_perc_{}".format(get('sampling_percent'), get('sampling'))
    
    common_path += "_{}".format(get('model_type')) + {
        ".*":        lambda: "",
        "pop_rec":   lambda: "",
        "bias_only": lambda: "",
        "MF_dot":    lambda: "_latent_size_{}_dropout_{}".format(get('latent_size'), get('dropout')),
        "MF":        lambda: "_latent_size_{}_dropout_{}".format(get('latent_size'), get('dropout')),
        "NeuMF":     lambda: "_latent_size_{}_dropout_{}".format(get('latent_size'), get('dropout')),
        "MVAE":      lambda: "_latent_size_{}_dropout_{}".format(get('latent_size'), get('dropout')),
        "SVAE":      lambda: "_latent_size_{}_dropout_{}_next_{}".format(get('latent_size'), get('dropout'), get('num_next')),
        "SASRec":    lambda: "_latent_size_{}_dropout_{}_heads_{}_blocks_{}".format(get('latent_size'), get('dropout'), get('num_heads'), get('num_blocks')),
    }[get('model_type')]() # lambda to ensure lazy evaluation

    if get('task') in [ 'implicit', 'sequential' ]:
        common_path += "_trn_negs_{}_tst_negs_{}".format(get('num_train_negs'), get('num_test_negs'))

    common_path += "_wd_{}_lr_{}".format(get('weight_decay'), get('lr'))

    return common_path

def remap_items(data):
    item_map = {}
    for user_data in data:
        for item, rating, time in user_data:
            if item not in item_map: item_map[item] = len(item_map) + 1

    for u in range(len(data)):
        data[u] = list(map(lambda x: [ item_map[x[0]], x[1], x[2] ], data[u]))

    return data

def file_write(log_file, s, dont_print=False):
    if dont_print == False: print(s)
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()

def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')

def log_end_epoch(hyper_params, metrics, epoch, time_elpased, metrics_on = '(VAL)', dont_print = False):
    string2 = ""
    for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
    string2 += ' ' + metrics_on

    ss  = '-' * 89
    ss += '\n| end of epoch {} | time = {:5.2f}'.format(epoch, time_elpased)
    ss += string2
    ss += '\n'
    ss += '-' * 89
    file_write(hyper_params['log_file'], ss, dont_print = dont_print)
