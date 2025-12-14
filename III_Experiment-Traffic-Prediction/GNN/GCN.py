import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GraphConv, BatchNorm, global_mean_pool
from torch_geometric.data import DataLoader
import argparse
import random
import logging
from datetime import datetime
from texttable import Texttable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def mape_loss(pred, target, epsilon=1e-8):
    pred = pred.view(-1)
    target = target.float().view(-1)
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=64, num_outputs=1):
        super(GCN, self).__init__()
        self.node_emb = nn.Linear(in_channels, hid_channels)
        self.hid_channels = hid_channels

        self.conv1 = GraphConv(in_channels=hid_channels, out_channels=hid_channels)
        self.bn1 = BatchNorm(hid_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = GraphConv(in_channels=hid_channels, out_channels=hid_channels)
        self.bn2 = BatchNorm(hid_channels)
        self.relu2 = nn.ReLU()

        self.predictor = nn.Sequential(
            nn.Linear(hid_channels, 2 * hid_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hid_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, num_outputs)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr)
        graph_x = global_mean_pool(node_x, batch)
        out = self.predictor(graph_x) 
        return out

    def get_node_reps(self, x, edge_index, edge_attr):
        edge_weight = None
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_weight = edge_attr
            elif edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                edge_weight = edge_attr.squeeze(-1)
            else:
                edge_weight = None

        if x.size(1) != self.hid_channels:
            x = self.node_emb(x)

        if edge_weight is not None and edge_weight.size(0) == edge_index.size(1):
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = self.relu1(self.bn1(x))

        if edge_weight is not None and edge_weight.size(0) == edge_index.size(1):
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
    train_data = torch.load(osp.join(raw_dir, 'train.pt'), weights_only=False)
    val_data = torch.load(osp.join(raw_dir, 'val.pt'), weights_only=False)
    test_data = torch.load(osp.join(raw_dir, 'test.pt'), weights_only=False)
    return train_data, val_data, test_data


######################################################################
# MAIN
######################################################################


def main():
    parser = argparse.ArgumentParser(description='Traffic')
    parser.add_argument('--cuda', default=3, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=1000, type=int, help='training iterations')
    parser.add_argument('--seed', nargs='?', default='[42]', help='random seed')
    parser.add_argument('--channels', default=64, type=int, help='width of network')
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
        attr_results = {'train_mape': [], 'val_mape': [], 'test_mape': []}

        train_data_list, val_data_list, test_data_list = load_new_dataset(data_base_dir, attribute_name)
        train_loader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)

        sample_data = next(iter(train_loader))
        in_channels = sample_data.x.size(1)
        experiment_name = f'{attribute_name}'
        exp_dir = osp.join('GCN_results', experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
        logger.info(f"\nModel training started for property: {attribute_name} (MAPE metric)")

        args_print(args, logger)
        for seed in args.seed:
            set_seed(seed)
            logger.info(f"--- Starting training for {attribute_name} with seed {seed} ---")

            model = GCN(in_channels, args.channels, num_outputs=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.net_lr)

            def train_mode():
                model.train()

            def val_mode():
                model.eval()

            def eval_mape(loader, model):
                model.eval()
                total_mape = 0.0
                with torch.no_grad():
                    for graph in loader:
                        graph.to(device)
                        out = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                        mape = mape_loss(out, graph.y)
                        total_mape += mape.item() * graph.y.size(0)
                mape = total_mape / len(loader.dataset)
                return mape

            cnt, last_val_mape = 0, float('inf')
            best_val_mape = float('inf')

            for epoch in range(args.epoch):
                train_mode()
                total_loss = 0.0
                num_batches = 0

                # Training
                for graph in train_loader:
                    graph.to(device)
                    optimizer.zero_grad()

                    out = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                    loss = mape_loss(out, graph.y)

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                # Validation
                val_mode()
                with torch.no_grad():
                    train_mape = eval_mape(train_loader, model)
                    val_mape = eval_mape(val_loader, model)

                avg_loss = total_loss / num_batches

                logger.info(
                    f"Epoch [{epoch+1}/{args.epoch}] "
                    f"Train_MAPE: {train_mape:.6f}% Val_MAPE: {val_mape:.6f}% | "
                    f"Loss: {avg_loss:.6f}"
                )

                # Saving the best model
                if val_mape < best_val_mape:
                    best_val_mape = val_mape
                    torch.save(model.state_dict(), osp.join(exp_dir, f'predictor.pt'))
                    logger.info(f"Best model saved at epoch {epoch+1} with val_mape {val_mape:.6f}%")

                # Early stopping
                if epoch >= args.pretrain:
                    if val_mape >= last_val_mape:
                        cnt += 1
                    else:
                        cnt = 0
                        last_val_mape = val_mape
                    if cnt >= 100:
                        logger.info("Early Stopping")
                        break

        # Loading the best model
        model.load_state_dict(torch.load(osp.join(exp_dir, f'predictor.pt'), weights_only=False))

        # Evaluating the best model
        with torch.no_grad():
            best_train_mape = eval_mape(train_loader, model)
            best_val_mape = eval_mape(val_loader, model)
            best_test_mape = eval_mape(test_loader, model)

        logger.info(f"Best model Accuracy for {attribute_name}, seed {seed}:")
        logger.info(f"  Train MAPE: {best_train_mape:.6f}%")
        logger.info(f"  Val MAPE:   {best_val_mape:.6f}%")
        logger.info(f"  Test MAPE:  {best_test_mape:.6f}%")

        attr_results['train_mape'].append(best_train_mape)
        attr_results['val_mape'].append(best_val_mape)
        attr_results['test_mape'].append(best_test_mape)

        train_tensor = torch.tensor(attr_results['train_mape'])
        val_tensor = torch.tensor(attr_results['val_mape'])
        test_tensor = torch.tensor(attr_results['test_mape'])

        all_results[attribute_name] = {
            'train_mape': train_tensor,
            'val_mape': val_tensor,
            'test_mape': test_tensor
        }

        logger.info(f"\n=== Results for attribute {attribute_name} (MAPE) ===")
        result_summary = (
            f"Train MAPE: {train_tensor.mean():.6f}% "
            f"Val MAPE: {val_tensor.mean():.6f}% "
            f"Test MAPE: {test_tensor.mean():.6f}%"
        )
        logger.info(result_summary)


if __name__ == "__main__":
    main()
