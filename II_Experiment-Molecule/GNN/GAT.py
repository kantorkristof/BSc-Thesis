import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
from torch_geometric.data import DataLoader
import argparse
import random
import logging
from datetime import datetime
from texttable import Texttable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=64, num_outputs=1, heads=4):
        super(GAT, self).__init__()
        self.node_emb = nn.Linear(in_channels, hid_channels)
        self.hid_channels = hid_channels
        self.heads = heads
        
        self.conv1 = GATConv(in_channels=hid_channels, out_channels=hid_channels//heads, heads=heads, concat=True, dropout=0.1)
        self.bn1 = BatchNorm(hid_channels) 
        self.relu1 = nn.ReLU()
        
        self.conv2 = GATConv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, concat=False, dropout=0.1)
        self.bn2 = BatchNorm(hid_channels)  
        self.relu2 = nn.ReLU()

        self.predictor = nn.Sequential(
            nn.Linear(hid_channels, 2*hid_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, num_outputs)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr)
        graph_x = global_mean_pool(node_x, batch)
        out = self.predictor(graph_x)   
        return out

    def get_node_reps(self, x, edge_index, edge_attr):
        edge_weight = edge_attr.view(-1) if edge_attr is not None else None

        if x.size(1) != self.hid_channels:
            x = self.node_emb(x)

        if edge_weight is not None:
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = self.relu1(self.bn1(x))  
        
        if edge_weight is not None:
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv2(x, edge_index)
        x = self.relu2(self.bn2(x)) 
        
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info('\n' + table.draw())


class Logger:
    logger = None

    @staticmethod
    def get_logger(filename: str = None):
        if not Logger.logger:
            Logger.init_logger(filename=filename)
        return Logger.logger

    @staticmethod
    def init_logger(level=logging.INFO,
                    fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename: str = None):
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        fmt = logging.Formatter(fmt)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if filename and os.path.exists(filename):
            os.remove(filename)
        if filename:
            fh = logging.FileHandler(filename)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.setLevel(level)
        Logger.logger = logger
        return logger


def load_new_dataset(base_dir, attribute_name):
    raw_dir = osp.join(base_dir, attribute_name)
    train_data = torch.load(osp.join(raw_dir, 'train.pt'))
    val_data = torch.load(osp.join(raw_dir, 'val.pt'))
    test_data = torch.load(osp.join(raw_dir, 'test.pt'))
    return train_data, val_data, test_data


######################################################################
# MAIN
######################################################################


def main():
    parser = argparse.ArgumentParser(description='GAT for Graph Regression')
    parser.add_argument('--cuda', default=5, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=1000, type=int, help='training iterations')
    parser.add_argument('--seed', nargs='?', default='[42]', help='random seed')
    parser.add_argument('--channels', default=64, type=int, help='width of network')
    parser.add_argument('--heads', default=1, type=int, help='GAT attention heads')
    parser.add_argument('--pretrain', default=0, type=int, help='pretrain epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()
    args.seed = eval(args.seed)
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')

    data_base_dir = args.datadir
    available_attributes = []
    for item in os.listdir(data_base_dir):
        item_path = osp.join(data_base_dir, item)
        print('item_path:', item_path)
        if os.path.isdir(item_path):
            if all(os.path.exists(osp.join(item_path, fname)) for fname in ['train.pt', 'val.pt', 'test.pt']):
                available_attributes.append(item)

    all_results = {}
    print(available_attributes)

    for attribute_name in available_attributes:
        attr_results = {'train_mse': [], 'val_mse': [], 'test_mse': []}

        train_data_list, val_data_list, test_data_list = load_new_dataset(data_base_dir, attribute_name)
        train_loader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)

        sample_data = next(iter(train_loader))
        in_channels = sample_data.x.size(1)
        experiment_name = f'{attribute_name}'
        exp_dir = osp.join('GAT_results', experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
        logger.info(f"\nModel training started for property: {attribute_name}")

        args_print(args, logger)
        for seed in args.seed:
            set_seed(seed)
            logger.info(f"--- Starting training for {attribute_name} with seed {seed} ---")

            model = GAT(in_channels, args.channels, num_outputs=1, heads=args.heads).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.net_lr)
            criterion = nn.MSELoss()

            def train_mode():
                model.train()

            def val_mode():
                model.eval()

            def eval_mse(loader, model):
                model.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for graph in loader:
                        graph.to(device)
                        out = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                        loss = criterion(out.view(-1), graph.y.float())
                        total_loss += loss.item() * graph.y.size(0)
                mse = total_loss / len(loader.dataset)
                return mse

            cnt, last_val_mse = 0, float('inf')
            best_val_mse = float('inf')

            for epoch in range(args.epoch):
                train_mode()
                total_loss = 0.0
                num_batches = 0

                # Training
                for graph in train_loader:
                    graph.to(device)
                    optimizer.zero_grad()

                    out = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                    loss = criterion(out.view(-1), graph.y.float())

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                # Validation
                val_mode()
                with torch.no_grad():
                    train_mse = eval_mse(train_loader, model)
                    val_mse = eval_mse(val_loader, model)

                avg_loss = total_loss / num_batches

                logger.info(
                    f"Epoch [{epoch+1}/{args.epoch}] "
                    f"Train_MSE: {train_mse:.6f} Val_MSE: {val_mse:.6f} | "
                    f"Loss: {avg_loss:.6f}"
                )

                # Saving the best model
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    torch.save(model.state_dict(), osp.join(exp_dir, f'predictor.pt'))
                    logger.info(f"Best model saved at epoch {epoch+1} with val_mse {val_mse:.6f}")

                # Early stopping
                if epoch >= args.pretrain:
                    if val_mse >= last_val_mse:
                        cnt += 1
                    else:
                        cnt = 0
                        last_val_mse = val_mse
                    if cnt >= 100:
                        logger.info("Early Stopping")
                        break

            # Loading the best model
            model.load_state_dict(torch.load(osp.join(exp_dir, f'predictor.pt')))

            # Evaluating the best model
            with torch.no_grad():
                best_train_mse = eval_mse(train_loader, model)
                best_val_mse = eval_mse(val_loader, model)
                best_test_mse = eval_mse(test_loader, model)

            logger.info(f"Best model Accuracy for {attribute_name}, seed {seed}:")
            logger.info(f"  Train MSE: {best_train_mse:.6f}")
            logger.info(f"  Val MSE:   {best_val_mse:.6f}")
            logger.info(f"  Test MSE:  {best_test_mse:.6f}")

            attr_results['train_mse'].append(best_train_mse)
            attr_results['val_mse'].append(best_val_mse)
            attr_results['test_mse'].append(best_test_mse)

        train_tensor = torch.tensor(attr_results['train_mse'])
        val_tensor = torch.tensor(attr_results['val_mse'])
        test_tensor = torch.tensor(attr_results['test_mse'])

        all_results[attribute_name] = {
            'train_mse': train_tensor,
            'val_mse': val_tensor,
            'test_mse': test_tensor
        }

        logger.info(f"\n=== Results for attribute {attribute_name} ===")
        result_summary = (
            f"Train MSE: {train_tensor.mean():.6f}  "
            f"Val MSE: {val_tensor.mean():.6f}  "
            f"Test MSE: {test_tensor.mean():.6f}"
        )
        logger.info(result_summary)


if __name__ == "__main__":
    main()
