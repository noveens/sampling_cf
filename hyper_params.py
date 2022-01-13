hyper_params = {
    ## Dataset
    # [ 'ml-100k', 'magazine', 'software', 'luxury', 'fashion', 'industrial', 'goodreads_comics' ]
    'dataset':              'magazine',
    
    ## Tasks
    # [ 'explicit', 'implicit', 'sequential' ]
    'task':                 'sequential',

    ## Sampling
    # [ complete_data, user_rns, item_rns, interaction_rns, temporal, tail_user_remove, svp_bias_only, svp_MF_dot ]
    # [ pagerank, random_walk, forest_fire ]
    # [ 20, 40, 60, 80, 90, 99 ]
    'sampling':             'complete_data', 
    'sampling_percent':     40, # Only relevant if sampling != "complete_data" 

    ## Sampling -- SVP
    # [ forgetting_events, forgetting_events_propensity, forgetting_events_user, forgetting_events_user_propensity ]
    'sampling_svp':         'forgetting_events_user', # Only relevant if hyper_params['sampling'] == "svp_*"

    ## Models 
    # [ 'pop_rec', 'bias_only', 'MF', 'MF_dot', 'NeuMF', 'MVAE' ]
    # [ 'TransRec', 'SVAE', 'SASRec' ]
    'model_type':           'SASRec', 
    'num_train_negs':       1,
    'num_test_negs':        100,

    ## All methods
    'latent_size':          32, 
    'dropout':              0.5,
    
    ## SASRec
    'num_heads':            1,
    'num_blocks':           2,

    ## SVAE
    'num_next':             2,
}
