import torch_geometric
import torch
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, hidden_size=(16,), dropout=0.3, df=None):
        super().__init__()
        self.dropout = dropout
        self.conv_layers = []
        n = len(hidden_size)
        for i in range(n):
            n_input = df.num_node_features if i == 0 else hidden_size[i - 1]
            n_output = hidden_size[i]
            self.conv_layers.append(torch_geometric.nn.GCNConv(n_input, n_output, improved=True))
        self.conv_layers = torch.nn.ParameterList(self.conv_layers)
        self.fc = torch.nn.Linear(hidden_size[-1], 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index.long(), data.edge_attr
        for layer in self.conv_layers:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index, edge_weight)
            x = torch.nn.functional.relu(x).float()
            x
        x = self.fc(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_size=(16,), heads=(3,), dropout=0.3, df=None):
        super().__init__()
        self.conv_layers = []
        n = len(hidden_size)
        if len(heads) != n:
            heads = [heads[0]] * n

        for i in range(n):
            n_input = df.num_node_features if i == 0 else hidden_size[i - 1] * heads[i - 1]
            n_output = hidden_size[i]
            self.conv_layers.append(
                torch_geometric.nn.GATConv(
                    n_input,
                    n_output,
                    heads=heads[i],
                    dropout=dropout,
                )
            )
        self.fc = torch.nn.Linear(hidden_size[-1] * heads[-1], 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index.long(), data.edge_attr
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_weight)
            x
        x = self.fc(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, hidden_size=(16,), dropout=0.3, df=None):
        super().__init__()
        self.dropout = dropout
        self.conv_layers = []
        n = len(hidden_size)
        for i in range(n):
            n_input = df.num_node_features if i == 0 else hidden_size[i - 1]
            n_output = hidden_size[i]
            self.conv_layers.append(torch_geometric.nn.SAGEConv(n_input, n_output))
        self.conv_layers = torch.nn.ParameterList(self.conv_layers)
        self.fc = torch.nn.Linear(hidden_size[-1], 1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.long()
        for layer in self.conv_layers:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            x = torch.nn.functional.relu(x).float()
            x
        x = self.fc(x)
        return x


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = torch_geometric.nn.HeteroConv(
                {
                    ("crew", "in", "movies"): torch_geometric.nn.SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("cast", "in", "movies"): torch_geometric.nn.SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("movies", "rev_in", "crew"): torch_geometric.nn.SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("movies", "rev_in", "cast"): torch_geometric.nn.SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                },
                aggr="mean",
            )
            self.convs.append(conv)

        self.lin = torch_geometric.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {
                key: torch.nn.functional.dropout(x.to(torch.float64), p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["movies"].double()).double()
