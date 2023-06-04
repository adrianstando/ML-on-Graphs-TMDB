import torch_geometric
import torch
import numpy as np
import pandas as pd
import os

from models import HeteroGNN
from dataset import TMDBDataset


task_id = os.getenv("SLURM_ARRAY_TASK_ID")

hidden_sizes = [8, 16, 32]
dropouts = [0.0, 0.3, 0.5]
lrs = [0.1, 0.03, 0.01]

graph_features = ["overview", "keywords"]


def main(task_id):
    # global task_id
    global hidden_sizes
    global dropouts
    global lrs
    global graph_features

    task_id = int(task_id)
    task_id = task_id - 1
    print(f"Task id: {task_id}")

    hidden_sizes_idx = task_id // (len(dropouts) * len(lrs) * len(graph_features))
    dropout_idx = (task_id // (len(lrs) * len(graph_features))) % len(dropouts)
    lr_idx = (task_id // len(graph_features)) % len(lrs)

    graph_features_idx = task_id % len(graph_features)

    hidden_size = hidden_sizes[hidden_sizes_idx]
    dropout = dropouts[dropout_idx]
    lr = lrs[lr_idx]
    graph_features = graph_features[graph_features_idx]

    train_model(hidden_size, dropout, lr, graph_features)


def train_model(hidden_size, dropout, lr, graph_features):
    df = TMDBDataset(
        root="./tmp",
        node_feature_method="counter",
        node_feature_params={"min_df": 0.1 if graph_features == "overview" else 0.015},
        node_feature_column_source=graph_features,
        add_additional_node_features=True,
        edge_weight_column_source="cast",
        jaccard_distance_threshold=0,
        graph_type="heterogeneous",
    )

    graph = df[0]
    graph["movies"].y = np.log(graph["movies"].y)

    split = torch_geometric.transforms.RandomNodeSplit(num_val=0, num_test=0.2)
    graph = split(graph)

    model = HeteroGNN(hidden_channels=hidden_size, out_channels=1, num_layers=1, dropout=dropout)
    model.to(torch.float64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_mse = []
    test_mse = []

    for epoch in range(250):
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        train_mask = graph["movies"].train_mask
        loss = loss_fn(out[train_mask].ravel(), graph["movies"].y[train_mask])
        loss.backward()
        optimizer.step()
        train_mse.append(loss.item())

        test_mask = graph["movies"].test_mask
        test_mse_val = loss_fn(out[test_mask].ravel(), graph["movies"].y[test_mask])
        test_mse.append(test_mse_val)

    pd.DataFrame(
        {
            "architecture": hidden_size,
            "dropout": dropout,
            "lr": lr,
            "node_features": graph_features,
            "train_mse": train_mse,
            "test_mse": test_mse,
        }
    ).to_csv(
        f"results/hetero_nf={graph_features}_size={hidden_size}_dropout={dropout}_lr={lr}.csv",
        index=False,
    )


if __name__ == "__main__":
    main(1)
