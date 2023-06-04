import torch_geometric
import torch
import numpy as np
import pandas as pd
import os

from models import GCN, GAT, SAGE
from dataset import TMDBDataset


task_id = os.getenv("SLURM_ARRAY_TASK_ID")

hidden_sizes = [
    (20,),
    (32, 16),
    (64, 32, 32),
]
dropouts = [0.0, 0.3, 0.5]
lrs = [0.1, 0.03, 0.01]

graph_features = [
    ("overview", "cast"),
    ("overview", "crew"),
    ("overview", "keywords"),
    ("keywords", "cast"),
    ("keywords", "crew"),
]

models = {
    "GCN": GCN,
    "GAT": GAT,
    "SAGE": SAGE,
}


def main():
    global task_id
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

    train_models(hidden_size, dropout, lr, graph_features)


def train_models(hidden_size, dropout, lr, graph_features):
    global models

    df = TMDBDataset(
        root="./tmp",
        node_feature_method="counter",
        node_feature_params={"min_df": 0.1 if graph_features[0] == "overview" else 0.015},
        node_feature_column_source=graph_features[0],
        add_additional_node_features=True,
        edge_weight_column_source=graph_features[1],
        jaccard_distance_threshold=0,
        graph_type="homogenous",
    )

    graph = df[0]
    graph.y = np.log(graph.y)
    split = torch_geometric.transforms.RandomNodeSplit(num_val=0, num_test=0.2)
    graph = split(graph)

    for model_name, model_class in models.items():
        print(f"Training {model_name} with hidden size {hidden_size}, dropout {dropout}, lr {lr}")
        train_model(model_name, model_class, hidden_size, dropout, lr, df, graph, graph_features)


def train_model(model_name, model_class, hidden_size, dropout, lr, df, graph, graph_features):
    model = model_class(hidden_size=hidden_size, dropout=dropout, df=df)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_mse = []
    test_mse = []

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = loss_fn(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        train_mse.append(loss.item())

        model.eval()
        test_mse_val = loss_fn(out[graph.test_mask], graph.y[graph.test_mask]).item()
        test_mse.append(test_mse_val)

    pd.DataFrame(
        {
            "model": model_name,
            "architecture": str(hidden_size),
            "dropout": dropout,
            "lr": lr,
            "node_features": graph_features[0],
            "edge_features": graph_features[1],
            "train_mse": train_mse,
            "test_mse": test_mse,
        }
    ).to_csv(
        f"results/nf={graph_features[0]}_ef={graph_features[1]}_model={model_name}{str(hidden_size)}_dropout={dropout}_lr={lr}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
