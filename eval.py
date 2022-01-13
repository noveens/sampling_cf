import torch
import numpy as np
from numba import jit, float32, float64, int64

from utils import INF

def evaluate(model, criterion, reader, hyper_params, item_propensity, topk = [ 10, 100 ], test = False):
    metrics = {}

    # Do a negative sampled item-space evaluation (only on the validation set)
    # if the dataset is too big 
    partial_eval = (not test) and hyper_params['partial_eval'] 
    partial_eval = partial_eval and (hyper_params['model_type'] not in [ 'MVAE', 'SVAE', 'pop_rec' ])
    if partial_eval: metrics['eval'] = 'partial'

    if hyper_params['task'] == 'explicit': metrics['MSE'] = 0.0
    else:
        preds, y_binary = [], []
        for kind in [ 'HR', 'NDCG', 'PSP' ]:
            for k in topk: 
                metrics['{}@{}'.format(kind, k)] = 0.0

    model.eval()
    with torch.no_grad():
        for data, y in reader:
            output = model(data, eval = True)
            if hyper_params['model_type'] in [ 'MVAE', 'SVAE' ]: output, _, _ = output
            if hyper_params['model_type'] == 'SVAE': output = output[:, -1, :]

            if hyper_params['task'] == 'explicit': 
                metrics['MSE'] += torch.sum(criterion(output, y, return_mean = False).data)
            else:
                function = evaluate_batch_partial if partial_eval else evaluate_batch

                metrics, temp_preds, temp_y = function(data, output, y, item_propensity, topk, metrics)
                preds += temp_preds
                y_binary += temp_y

    if hyper_params['task'] == 'explicit':
        metrics['MSE'] = round(float(metrics['MSE']) / reader.num_interactions, 4)
    else:
        # NOTE: sklearn's `roc_auc_score` is suuuuper slow
        metrics['AUC'] = round(fast_auc(np.array(y_binary), np.array(preds)), 4)
        
        for kind in [ 'HR', 'NDCG', 'PSP' ]:
            for k in topk: 
                metrics['{}@{}'.format(kind, k)] = round(
                    float(100.0 * metrics['{}@{}'.format(kind, k)]) / reader.num_interactions, 4
                )

    return metrics

def evaluate_batch(data, output_batch, y, item_propensity, topk, metrics):
    # Y
    train_positive, test_positive_set = y

    # Data
    _, _, auc_negatives = data

    # AUC Stuff
    temp_preds, temp_y = [], []
    logits_cpu = output_batch.cpu().numpy()
    for b in range(len(output_batch)):
        # Validation set could have 0 positive interactions
        if len(test_positive_set[b]) == 0: continue

        temp_preds += np.take(logits_cpu[b], np.array(list(test_positive_set[b]))).tolist()
        temp_y += [ 1.0 for _ in range(len(test_positive_set[b])) ]

        temp_preds += np.take(logits_cpu[b], auc_negatives[b]).tolist()
        temp_y += [ 0.0 for _ in range(len(auc_negatives[b])) ]

    # Marking train-set consumed items as negative INF
    for b in range(len(output_batch)): output_batch[b][ train_positive[b] ] = -INF

    _, indices = torch.topk(output_batch, min(item_propensity.shape[0], max(topk)), sorted = True)
    indices = indices.cpu().numpy().tolist()

    for k in topk: 
        for b in range(len(output_batch)):
            num_pos = float(len(test_positive_set[b]))
            # Validation set could have 0 positive interactions after sampling
            if num_pos == 0: continue

            metrics['HR@{}'.format(k)] += float(len(set(indices[b][:k]) & test_positive_set[b])) / float(min(num_pos, k))

            test_positive_sorted_psp = sorted([ item_propensity[x] for x in test_positive_set[b] ])[::-1]

            dcg, idcg, psp, max_psp = 0.0, 0.0, 0.0, 0.0
            for at, pred in enumerate(indices[b][:k]):
                if pred in test_positive_set[b]: 
                    dcg += 1.0 / np.log2(at + 2)
                    psp += float(item_propensity[pred]) / float(min(num_pos, k))
                if at < num_pos: 
                    idcg += 1.0 / np.log2(at + 2)
                    max_psp += test_positive_sorted_psp[at]

            metrics['NDCG@{}'.format(k)] += dcg / idcg
            metrics['PSP@{}'.format(k)] += psp / max_psp

    return metrics, temp_preds, temp_y

def evaluate_batch_partial(data, output, y, item_propensity, topk, metrics):
    _, test_pos_items, _ = data
    test_pos_items = test_pos_items.cpu().numpy()

    pos_score, neg_score = output
    pos_score, neg_score = pos_score.cpu().numpy(), neg_score.cpu().numpy()

    temp_preds, temp_y, hr, ndcg, psp = evaluate_batch_partial_jit(
        pos_score, neg_score, test_pos_items, np.array(item_propensity), np.array(topk)
    )

    for at_k, k in enumerate(topk): 
        metrics['HR@{}'.format(k)] += hr[at_k]
        metrics['NDCG@{}'.format(k)] += ndcg[at_k]
        metrics['PSP@{}'.format(k)] += psp[at_k]

    return metrics, temp_preds.tolist(), temp_y.tolist()

@jit('Tuple((float32[:], float32[:], float32[:], float32[:], float32[:]))(float32[:,:], float32[:,:], int64[:,:], float64[:], int64[:])')
def evaluate_batch_partial_jit(pos_score, neg_score, test_pos_items, item_propensity, topk):
    temp_preds = np.zeros(
        ((pos_score.shape[0] * pos_score.shape[1]) + (neg_score.shape[0] * neg_score.shape[1])), 
        dtype = np.float32
    )
    temp_y = np.zeros(temp_preds.shape, dtype = np.float32)
    at_preds = 0

    hr_arr = np.zeros((len(topk)), dtype = np.float32)
    ndcg_arr = np.zeros((len(topk)), dtype = np.float32)
    psp_arr = np.zeros((len(topk)), dtype = np.float32)

    for b in range(len(pos_score)):
        pos, neg = pos_score[b, :], neg_score[b, :]

        # pos will be padded, un-pad it
        last_index = len(pos) - 1
        while last_index > 0 and pos[last_index] == pos[last_index - 1]: last_index -= 1
        pos = pos[:last_index + 1]

        # Add to AUC
        temp_preds[at_preds:at_preds+len(pos)] = pos
        temp_y[at_preds:at_preds+len(pos)] = 1
        at_preds += len(pos)

        temp_preds[at_preds:at_preds+len(neg)] = neg
        temp_y[at_preds:at_preds+len(neg)] = 0
        at_preds += len(neg)

        # get rank of all elements in pos
        temp_ranks = np.argsort(- np.concatenate((pos, neg)))

        # To maintain order
        pos_ranks = np.zeros(len(pos))
        for at, r in enumerate(temp_ranks):
            if r < len(pos): pos_ranks[r] = at + 1

        test_positive_sorted_psp = sorted([ item_propensity[x] for x in test_pos_items[b] ])[::-1]

        for at_k, k in enumerate(topk): 
            num_pos = float(len(pos))
            
            hr_arr[at_k] += np.sum(pos_ranks <= k) / float(min(num_pos, k))

            dcg, idcg, psp, max_psp = 0.0, 0.0, 0.0, 0.0
            for at, rank in enumerate(pos_ranks):
                if rank <= k:
                    dcg += 1.0 / np.log2(rank + 1) # 1-based indexing
                    psp += item_propensity[test_pos_items[b][at]] / float(min(num_pos, k))
                idcg += 1.0 / np.log2(at + 2)
                max_psp += test_positive_sorted_psp[at]
            
            ndcg_arr[at_k] += dcg / idcg
            psp_arr[at_k] += psp / max_psp

    return temp_preds[:at_preds], temp_y[:at_preds], hr_arr, ndcg_arr, psp_arr

@jit(float64(float64[:], float64[:]))
def fast_auc(y_true, y_prob):
    y_true = y_true[np.argsort(y_prob)]
    nfalse, auc = 0, 0
    for i in range(len(y_true)):
        nfalse += (1 - y_true[i])
        auc += y_true[i] * nfalse
    return auc / (nfalse * (len(y_true) - nfalse))