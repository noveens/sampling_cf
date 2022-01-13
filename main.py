import os
import time
import importlib
import datetime as dt
from tqdm import tqdm

from utils import file_write, log_end_epoch, INF, valid_hyper_params
from data_path_constants import get_log_file_path, get_model_file_path

# NOTE: No global-level torch imports as the GPU-ID is set through code

def train(model, criterion, optimizer, reader, hyper_params, forgetting_events, track_events):
    import torch

    model.train()
    
    # Initializing metrics since we will calculate MSE on the train set on the fly
    metrics = {}
    
    # Initializations
    at = 0
    
    # Train for one epoch, batch-by-batch
    loop = tqdm(reader)
    for data, y in loop:
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
    
        # Forward pass
        output = model(data)

        # Compute per-interaction loss
        loss = criterion(output, y, return_mean = False)
        criterion.anneal(1.0 / float(len(reader) * hyper_params['epochs']))

        # loop.set_description("Loss: {}".format(round(float(loss), 4)))
        
        # Track forgetting events
        if track_events:
            with torch.no_grad():
                if hyper_params['task'] == 'explicit': forgetting_events[at : at+data[0].shape[0]] += loss.data
                else:
                    pos_output, neg_output = output
                    pos_output = pos_output.repeat(1, neg_output.shape[1])
                    num_incorrect = torch.sum((neg_output > pos_output).float(), -1)
                    forgetting_events[at : at+data[0].shape[0]] += num_incorrect.data
                    
                at += data[0].shape[0]

        # Backward pass
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

    return metrics, forgetting_events

def train_complete(hyper_params, train_reader, val_reader, model, model_class, track_events):
    import torch

    from loss import CustomLoss
    from eval import evaluate
    from torch_utils import is_cuda_available

    criterion = CustomLoss(hyper_params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyper_params['lr'], betas=(0.9, 0.98),
        weight_decay=hyper_params['weight_decay']
    )

    file_write(hyper_params['log_file'], str(model))
    file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

    try:
        best_MSE = float(INF)
        best_AUC = -float(INF)
        best_HR = -float(INF)
        decreasing_streak = 0
        forgetting_events = None
        if track_events: 
            forgetting_events = torch.zeros(train_reader.num_interactions).float()
            if is_cuda_available: forgetting_events = forgetting_events.cuda()

        for epoch in range(1, hyper_params['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training for one epoch
            metrics, local_forgetted_count = train(
                model, criterion, optimizer, train_reader, hyper_params, 
                forgetting_events, track_events
            )

            # Calulating the metrics on the validation set
            if (epoch % hyper_params['validate_every'] == 0) or (epoch == 1):
                metrics = evaluate(model, criterion, val_reader, hyper_params, train_reader.item_propensity)
                metrics['dataset'] = hyper_params['dataset']
                decreasing_streak += 1

                # Save best model on validation set
                if hyper_params['task'] == 'explicit' and metrics['MSE'] < best_MSE:
                    print("Saving model...")
                    torch.save(model.state_dict(), hyper_params['model_path'])
                    decreasing_streak, best_MSE = 0, metrics['MSE']
                elif hyper_params['task'] != 'explicit' and metrics['AUC'] > best_AUC:
                    print("Saving model...")
                    torch.save(model.state_dict(), hyper_params['model_path'])
                    decreasing_streak, best_AUC = 0, metrics['AUC']
                elif hyper_params['task'] != 'explicit' and metrics['HR@10'] > best_HR:
                    print("Saving model...")
                    torch.save(model.state_dict(), hyper_params['model_path'])
                    decreasing_streak, best_HR = 0, metrics['HR@10']
            
            log_end_epoch(hyper_params, metrics, epoch, time.time() - epoch_start_time, metrics_on = '(VAL)')

            # Check if need to early-stop
            if 'early_stop' in hyper_params and decreasing_streak >= hyper_params['early_stop']:
                file_write(hyper_params['log_file'], "Early stopping..")
                break
            
    except KeyboardInterrupt: print('Exiting from training early')

    # Load best model and return it for evaluation on test-set
    if os.path.exists(hyper_params['model_path']):
        model = model_class(hyper_params)
        if is_cuda_available: model = model.cuda()
        model.load_state_dict(torch.load(hyper_params['model_path']))
    
    model.eval()

    if track_events: forgetting_events = forgetting_events.cpu().numpy() / float(hyper_params['epochs'])

    return model, forgetting_events

def train_neumf(hyper_params, train_reader, val_reader, track_events):
    from pytorch_models.NeuMF import GMF, MLP, NeuMF
    from torch_utils import is_cuda_available, xavier_init

    initial_path = hyper_params['model_path']

    # Pre-Training the GMF Model
    hyper_params['model_path'] = initial_path[:-3] + "_gmf.pt"
    gmf_model = GMF(hyper_params)
    if is_cuda_available: gmf_model = gmf_model.cuda()
    xavier_init(gmf_model)
    gmf_model, _ = train_complete(hyper_params, train_reader, val_reader, gmf_model, GMF, track_events)

    # Pre-Training the MLP Model
    hyper_params['model_path'] = initial_path[:-3] + "_mlp.pt"
    mlp_model = MLP(hyper_params)
    if is_cuda_available: mlp_model = mlp_model.cuda()
    xavier_init(mlp_model)
    mlp_model, _ = train_complete(hyper_params, train_reader, val_reader, mlp_model, MLP, track_events)

    # Training the final NeuMF Model
    hyper_params['model_path'] = initial_path
    model = NeuMF(hyper_params)
    if is_cuda_available: model = model.cuda()
    model.init(gmf_model, mlp_model)
    model, forgetting_events = train_complete(hyper_params, train_reader, val_reader, model, NeuMF, track_events)

    # Remove GMF and MLP models
    mlp_path = initial_path[:-3] + "_mlp.pt"
    gmf_path = initial_path[:-3] + "_gmf.pt"
    os.remove(mlp_path) ; os.remove(gmf_path)
    
    return model, forgetting_events

def main_pytorch(hyper_params, track_events = False, eval_full = True):
    from load_data import load_data
    from eval import evaluate
    
    from torch_utils import is_cuda_available, xavier_init, get_model_class
    from loss import CustomLoss

    if not valid_hyper_params(hyper_params): 
        print("Invalid task combination specified, exiting.")
        return

    # Load the data readers
    train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params, track_events = track_events)
    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(len(train_reader)))
    file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(len(val_reader)))
    file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(len(test_reader)))

    # Initialize & train the model
    start_time = time.time()

    if hyper_params['model_type'] == 'NeuMF': 
        model, forgetting_events = train_neumf(hyper_params, train_reader, val_reader, track_events)
    else:
        model = get_model_class(hyper_params)(hyper_params)
        if is_cuda_available: model = model.cuda()
        xavier_init(model)
        model, forgetting_events = train_complete(
            hyper_params, train_reader, val_reader, model, get_model_class(hyper_params), track_events
        )

    metrics = {}
    if eval_full:
        # Calculating MSE on test-set
        criterion = CustomLoss(hyper_params)
        metrics = evaluate(model, criterion, test_reader, hyper_params, train_reader.item_propensity, test = True)
        log_end_epoch(hyper_params, metrics, 'final', time.time() - start_time, metrics_on = '(TEST)')

    # We have no space left for storing the models
    os.remove(hyper_params['model_path'])
    del model, train_reader, test_reader, val_reader
    return metrics, forgetting_events

def main_pop_rec(hyper_params):
    from load_data import load_data
    from eval import evaluate
    from loss import CustomLoss
    from pytorch_models.pop_rec import PopRec

    # Load the data readers
    train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)
    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of test batches: {:4d}\n\n".format(len(test_reader)))

    # Make the model
    start_time = time.time()
    model = PopRec(hyper_params, train_reader.get_item_count_map())

    # Calculating MSE on test-set
    criterion = CustomLoss(hyper_params)
    metrics = evaluate(model, criterion, test_reader, hyper_params, train_reader.item_propensity, test = True)

    log_end_epoch(hyper_params, metrics, 'final', time.time() - start_time, metrics_on = '(TEST)')
    
    del model, train_reader, test_reader, val_reader
    return metrics, None

def main(hyper_params, gpu_id = None): 

    if not valid_hyper_params(hyper_params): 
        print("Invalid task combination specified, exiting.")
        return

    # Setting GPU ID for running entire code ## Very Very Imp.
    if gpu_id is not None: 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # torch.cuda.set_device(int(gpu_id))
        # torch.cuda.empty_cache()

    # Set dataset specific hyper-params
    hyper_params.update(
        importlib.import_module('data_hyperparams.{}'.format(hyper_params['dataset'])).hyper_params
    )

    # Learning rate is "highly" dataset AND model specific
    if 'lr' not in hyper_params:
        if hyper_params['model_type'] == 'SASRec': hyper_params['lr'] = 0.006
        elif hyper_params['model_type'] == 'SVAE': hyper_params['lr'] = 0.02
        elif hyper_params['model_type'] == 'MVAE': hyper_params['lr'] = 0.01
        else: hyper_params['lr'] = 0.008

    hyper_params['log_file'] = get_log_file_path(hyper_params)
    hyper_params['model_path'] = get_model_file_path(hyper_params)

    if hyper_params['model_type'] == 'pop_rec': main_pop_rec(hyper_params)
    else: main_pytorch(hyper_params)

    # torch.cuda.empty_cache()

if __name__ == '__main__':
    from hyper_params import hyper_params
    main(hyper_params)
