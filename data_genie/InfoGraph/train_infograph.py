import dgl
import time
import argparse
import torch as th
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from data_genie.data_genie_utils import INFOGRAPH_MODEL_PATH
from data_genie.InfoGraph.infograph_model import InfoGraph
from data_genie.InfoGraph.infograph_dataset import SyntheticDataset

def argument():
    parser = argparse.ArgumentParser(description='InfoGraph')

    # training params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, default:-1, using CPU.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')

    # model params
    parser.add_argument('--n_layers', type=int, default=2, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimensionalities')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available(): args.device = 'cuda:{}'.format(args.gpu)
    else: args.device = 'cpu'

    return args
    
def collate(samples):
    ''' collate function for building graph dataloader'''
    graphs, labels = map(list, zip(*samples))

    # generate batched graphs
    batched_graph = dgl.batch(graphs)

    n_nodes = batched_graph.num_nodes()

    # generate graph_id for each node within the batch
    graph_id = th.zeros(n_nodes).long()
    N = 0
    id = 0
    for graph in graphs:
        N_next = N + graph.num_nodes()
        graph_id[N:N_next] = id
        N = N_next
        id += 1

    batched_graph.ndata['graph_id'] = graph_id

    return batched_graph, len(graphs)

def train_infograph(args):
    # Step 1: Prepare graph data   ===================================== #
    dataset = SyntheticDataset(feature_dimension = 32)
    print("Total # of graphs:", len(dataset.graphs), "/", dataset.orig_total, "\n")
    
    # creata dataloader for batch training
    dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size, collate_fn=collate,
        drop_last=False, shuffle=True
    )

    # Step 2: Create model =================================================================== #
    print("DIMENSION:", args.hid_dim, "; GCN Layers:", args.n_layers)
    model = InfoGraph(args.hid_dim, args.n_layers)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print('===== Before training ======')

    # Step 4: training epoches =============================================================== #
    best_loss = float(1e10)
    for epoch in range(1, args.epochs):
        loss_all = 0
        model.train()
        start_time = time.time()
    
        for graph, n_graph in tqdm(dataloader):
            graph = graph.to(args.device)
    
            optimizer.zero_grad()
            loss = model(graph)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * n_graph
    
        mean_loss = loss_all / len(dataloader)
        print('Epoch {}, Loss {:.4f}, Time {:.4f}'.format(epoch, mean_loss, time.time() - start_time))

        if mean_loss < best_loss:
            print("Saving...")
            th.save(model.state_dict(), INFOGRAPH_MODEL_PATH(args.hid_dim, args.n_layers))
            best_loss = mean_loss

if __name__ == '__main__':
    args = argument() ; print(args)
    train_infograph(args)
