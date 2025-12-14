import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GraphConv, BatchNorm, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import argparse
import random
import logging
from datetime import datetime
from torch_geometric.nn.conv import MessagePassing
from texttable import Texttable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Predictor(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=64, num_outputs=1, conv_unit=3):
        super(Predictor, self).__init__()
        self.node_emb = nn.Linear(in_channels, hid_channels)
        self.hid_channels = hid_channels
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GraphConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid_channels))
            self.relus.append(nn.ReLU())

        self.causal_mlp = nn.Sequential(
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
        logits = self.get_causal_pred(graph_x)
        return (logits > 0).float()

    def get_node_reps(self, x, edge_index, edge_attr):
        edge_weight = edge_attr.view(-1) if edge_attr is not None else None

        if x.size(1) != self.hid_channels:
            x = self.node_emb(x)

        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            if edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = relu(batch_norm(x))
        return x

    def get_causal_pred(self, causal_graph_x):
        pred = self.causal_mlp(causal_graph_x)
        return pred


class Rationale_Generator(nn.Module):
    def __init__(self, in_channels, causal_ratio, channels=64):
        super(Rationale_Generator, self).__init__()
        self.conv1 = GraphConv(in_channels=in_channels, out_channels=channels)
        self.conv2 = GraphConv(in_channels=channels, out_channels=channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels*2, channels*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels*4, 1)
        )
        self.ratio = causal_ratio

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (non_causal_edge_index, non_causal_edge_attr, non_causal_edge_weight) = split_graph(data, edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        non_causal_x, non_causal_edge_index, non_causal_batch, _ = relabel(x, non_causal_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
               (non_causal_x, non_causal_edge_index, non_causal_edge_attr, non_causal_edge_weight, non_causal_batch), \
               edge_score


######################################################################
# UTILITIES
######################################################################

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


def set_masks(mask, model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask


def clear_masks(model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])
    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def relabel(x, edge_index, batch, pos=None):
    if edge_index.numel() == 0:
        return x[:0], edge_index, batch[:0], pos
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def split_graph(data, edge_score, ratio):
    device = data.x.device
    causal_edge_index = torch.LongTensor([[], []]).to(device)
    causal_edge_weight = torch.tensor([]).to(device)
    causal_edge_attr = torch.tensor([]).to(device)
    non_causal_edge_index = torch.LongTensor([[], []]).to(device)
    non_causal_edge_weight = torch.tensor([]).to(device)
    non_causal_edge_attr = torch.tensor([]).to(device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = int(ratio * N)
        if data.edge_attr is not None:
            edge_attr = data.edge_attr[C:C+N]
        else:
            edge_attr = torch.ones((N, 1), device=device)
        single_mask = edge_score[C:C+N]
        single_mask_detach = single_mask.detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

        causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
        non_causal_edge_index = torch.cat([non_causal_edge_index, edge_index[:, idx_drop]], dim=1)

        causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
        non_causal_edge_weight = torch.cat([non_causal_edge_weight, -1 * single_mask[idx_drop]])

        causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
        non_causal_edge_attr = torch.cat([non_causal_edge_attr, edge_attr[idx_drop]])

    return (causal_edge_index, causal_edge_attr, causal_edge_weight), \
           (non_causal_edge_index, non_causal_edge_attr, non_causal_edge_weight)


def get_original_node_features(graph, edge_index):
    if edge_index.numel() == 0:
        return torch.empty(0, graph.x.size(1)).to(graph.x.device)
    sub_nodes = torch.unique(edge_index)
    return graph.x[sub_nodes]


def load_new_dataset(base_dir, attribute_name):
    raw_dir = osp.join(base_dir, attribute_name)
    train_data = torch.load(osp.join(raw_dir, 'train.pt'))
    val_data = torch.load(osp.join(raw_dir, 'val.pt'))
    test_data = torch.load(osp.join(raw_dir, 'test.pt'))
    return train_data, val_data, test_data


def compute_interventional_risk_variance(causal_pred_list, spurious_pred_list, labels):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    risks = []
    
    # Causal predictions risk
    for causal_pred in causal_pred_list:
        loss_per_sample = criterion(causal_pred.view(-1), labels.float())
        risk = loss_per_sample.mean()
        risks.append(risk)
    
    # Spurious predictions risk
    for spurious_pred in spurious_pred_list:
        loss_per_sample = criterion(spurious_pred.view(-1), labels.float())
        risk = loss_per_sample.mean()
        risks.append(risk)
    
    # If there is only one risk, the variance is 0
    if len(risks) <= 1:
        return torch.tensor(0.0).to(labels.device)
    
    risks_tensor = torch.stack(risks)
    variance = risks_tensor.var()
    
    return variance


def compute_dir_loss(causal_out, non_causal_out, labels, alpha):
    criterion = nn.BCEWithLogitsLoss()
    
    # R_causal: Causal loss
    causal_loss = criterion(causal_out.view(-1), labels.float())
    
    # R_spurious: Non-causal loss
    spurious_loss = criterion(non_causal_out.view(-1), labels.float())

    causal_pred_list = [causal_out]
    spurious_pred_list = [non_causal_out]
    dir_penalty = compute_interventional_risk_variance(causal_pred_list, spurious_pred_list, labels)
    
    total_loss = causal_loss + alpha * dir_penalty
    
    return total_loss, causal_loss, dir_penalty, spurious_loss


######################################################################
# MAIN
######################################################################

def main():
    parser = argparse.ArgumentParser(description='Training for Binary Classification with Causal Feature Learning')
    parser.add_argument('--cuda', default=2, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=1000, type=int, help='training iterations')
    parser.add_argument('--seed', nargs='?', default='[42]', help='random seed')
    parser.add_argument('--channels', default=64, type=int, help='width of network')
    parser.add_argument('--pretrain', default=0, type=int, help='pretrain epoch')
    parser.add_argument('--alpha', default=1e-0, type=float, help='invariant loss')
    parser.add_argument('--r', default=0.42, type=float, help='causal_ratio')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate for the predictor')
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
        attr_results = {'train_acc': [], 'val_acc': [], 'test_acc': []}

        train_data_list, val_data_list, test_data_list = load_new_dataset(data_base_dir, attribute_name)
        train_loader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data_list, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)

        sample_data = next(iter(train_loader))
        in_channels = sample_data.x.size(1)
        experiment_name = f'{attribute_name}'
        exp_dir = osp.join('DIR_results/', experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
        logger.info(f"\nModel training started for property: {attribute_name}")

        args_print(args, logger)
        for seed in args.seed:
            set_seed(seed)
            logger.info(f"--- Starting training for {attribute_name} with seed {seed} ---")
            g = Predictor(in_channels, args.channels, num_outputs=1).to(device)
            rationale_generator = Rationale_Generator(in_channels, args.r, args.channels).to(device)
            
            model_optimizer = torch.optim.Adam(
                list(g.parameters()) + list(rationale_generator.parameters()), 
                lr=args.net_lr
            )

            def train_mode():
                g.train()
                rationale_generator.train()

            def val_mode():
                g.eval()
                rationale_generator.eval()

            def test_metrics(loader, rationale_generator, predictor):
                correct = 0
                total = 0
                for graph in loader:
                    graph.to(device)
                    (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
                    (non_causal_x, non_causal_edge_index, non_causal_edge_attr, non_causal_edge_weight, non_causal_batch), edge_score = rationale_generator(graph)
                    
                    causal_x_orig = get_original_node_features(graph, causal_edge_index)
                    set_masks(causal_edge_weight, g)
                    out = predictor.get_causal_pred(
                        global_mean_pool(
                            g.get_node_reps(x=causal_x_orig, edge_index=causal_edge_index, edge_attr=causal_edge_attr), 
                            causal_batch
                        )
                    )
                    clear_masks(g)
                    predicted = (out > 0).float()
                    correct += (predicted.view(-1) == graph.y).sum().item()
                    total += graph.y.size(0)
                accuracy = correct / total
                return accuracy

            cnt, last_val_acc = 0, 0.0
            best_val_acc = 0.0
            
            for epoch in range(args.epoch):
                train_mode()
                epoch_total_loss = 0.0
                epoch_causal_loss = 0.0
                epoch_dir_penalty = 0.0
                epoch_spurious_loss = 0.0
                num_batches = 0
                
                for graph in train_loader:
                    graph.to(device)
                    model_optimizer.zero_grad()
                    
                    (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
                    (non_causal_x, non_causal_edge_index, non_causal_edge_attr, non_causal_edge_weight, non_causal_batch), edge_score = rationale_generator(graph)

                    causal_x_orig = get_original_node_features(graph, causal_edge_index)
                    non_causal_x_orig = get_original_node_features(graph, non_causal_edge_index)

                    set_masks(causal_edge_weight, g)
                    causal_rep = g.get_node_reps(x=causal_x_orig, edge_index=causal_edge_index, edge_attr=causal_edge_attr)
                    causal_out = g.get_causal_pred(global_mean_pool(causal_rep, causal_batch))
                    clear_masks(g)

                    set_masks(non_causal_edge_weight, g)
                    non_causal_rep = g.get_node_reps(x=non_causal_x_orig, edge_index=non_causal_edge_index, edge_attr=non_causal_edge_attr).detach()
                    non_causal_out = g.get_causal_pred(global_mean_pool(non_causal_rep, non_causal_batch))
                    clear_masks(g)

                    total_loss, causal_loss, dir_penalty, spurious_loss = compute_dir_loss(
                        causal_out, non_causal_out, graph.y, args.alpha
                    )
                    
                    # Backpropagation
                    total_loss.backward()
                    model_optimizer.step()
                    
                    epoch_total_loss += total_loss.item()
                    epoch_causal_loss += causal_loss.item()
                    epoch_dir_penalty += dir_penalty.item()
                    epoch_spurious_loss += spurious_loss.item()
                    num_batches += 1

                # Validation
                val_mode()
                with torch.no_grad():
                    train_acc = test_metrics(train_loader, rationale_generator, g)
                    val_acc = test_metrics(val_loader, rationale_generator, g)

                avg_total_loss = epoch_total_loss / num_batches
                avg_causal_loss = epoch_causal_loss / num_batches
                avg_dir_penalty = epoch_dir_penalty / num_batches
                avg_spurious_loss = epoch_spurious_loss / num_batches

                logger.info(
                    f"Epoch [{epoch+1}/{args.epoch}] "
                    f"Train_Acc: {train_acc:.4f} Val_Acc: {val_acc:.4f} | "
                    f"Total_Loss: {avg_total_loss:.4f} "
                    f"Causal_Loss: {avg_causal_loss:.4f} "
                    f"DIR_Penalty: {avg_dir_penalty:.6f} "
                    f"Spurious_Loss: {avg_spurious_loss:.4f}"
                )

                # Saving the best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(g.state_dict(), osp.join(exp_dir, f'predictor.pt'))
                    torch.save(rationale_generator.state_dict(), osp.join(exp_dir, f'rationale_generator.pt'))
                    logger.info(f"Best model saved at epoch {epoch+1} with val_acc {val_acc:.4f}")

                # Early stopping Mechanism
                if epoch >= args.pretrain:
                    if val_acc <= last_val_acc:
                        cnt += 1
                    else:
                        cnt = 0
                        last_val_acc = val_acc
                    if val_acc == 1.0:
                        logger.info("Validation accuracy reached 1.0, stopping training.")
                        break
                    if cnt >= 100:
                        logger.info("Early Stopping")
                        break

            # Loading the best model
            g.load_state_dict(torch.load(osp.join(exp_dir, f'predictor.pt')))
            rationale_generator.load_state_dict(torch.load(osp.join(exp_dir, f'rationale_generator.pt')))

            # Evaluating the best model
            g.eval()
            rationale_generator.eval()
            with torch.no_grad():
                best_train_acc = test_metrics(train_loader, rationale_generator, g)
                best_val_acc = test_metrics(val_loader, rationale_generator, g)
                best_test_acc = test_metrics(test_loader, rationale_generator, g)

            logger.info(f"Best model Accuracy for {attribute_name}, seed {seed}:")
            logger.info(f"  Train Accuracy: {best_train_acc:.4f}")
            logger.info(f"  Val Accuracy: {best_val_acc:.4f}")
            logger.info(f"  Test Accuracy: {best_test_acc:.4f}")

            attr_results['train_acc'].append(best_train_acc)
            attr_results['val_acc'].append(best_val_acc)
            attr_results['test_acc'].append(best_test_acc)

        train_tensor = torch.tensor(attr_results['train_acc'])
        val_tensor = torch.tensor(attr_results['val_acc'])
        test_tensor = torch.tensor(attr_results['test_acc'])

        all_results[attribute_name] = {
            'train_acc': train_tensor, 
            'val_acc': val_tensor, 
            'test_acc': test_tensor
        }

        logger.info(f"\n=== Results for attribute {attribute_name} ===")
        result_summary = (
            f"Train Accuracy: {train_tensor.mean():.4f}  "
            f"Val Accuracy: {val_tensor.mean():.4f}  "
            f"Test Accuracy: {test_tensor.mean():.4f}"
        )
        logger.info(result_summary)


if __name__ == "__main__":
    main()