from typing import Literal, Dict, Optional

import torch
import torch_geometric
import kaggle
import pandas as pd
import numpy as np
import os
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TMDBDataset(torch_geometric.data.InMemoryDataset):
    """
    The Graph Dataset representing the TMDB data.
    """

    def __init__(
        self,
        root: str = "./tmp",
        node_feature_method: Literal["counter", "tf-idf"] = "counter",
        node_feature_params: Dict = None,
        node_feature_column_source: Literal["overview", "keywords"] = "overview",
        edge_weight_column_source: Literal["cast", "crew", "keywords"] = "cast",
        jaccard_distance_threshold: float = 0,
    ):
        """
        The Graph Dataset representing the TMDB data.

        :param root: Root directory where the dataset should be saved.
        :param node_feature_method: Method to extract node features.
        :param node_feature_params: Dictionary with parameters for node feature extraction method.
        :param node_feature_column_source: The column from which the node features should be extracted.
        :param edge_weight_column_source: The column from which the weight features should be extracted.
        :param jaccard_distance_threshold: The Jaccard distance threshold above which the edges are added to the dataset
        """
        self.node_feature_method = node_feature_method
        self.node_feature_column_source = node_feature_column_source
        if node_feature_params is None:
            self.node_feature_params = {}
        else:
            self.node_feature_params = node_feature_params
        self.edge_weight_column_source = edge_weight_column_source
        self.jaccard_distance_threshold = jaccard_distance_threshold

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["tmdb_5000_credits_processed.csv", "tmdb_5000_movies_processed.csv"]

    @property
    def processed_file_names(self):
        return f"data_{self.node_feature_method}_{self.node_feature_column_source}_{self.edge_weight_column_source}.pt"

    def process(self):
        nodes, edges, edge_attributes, y = self._load_data_and_preprocess()
        data_graph = torch_geometric.data.Data(x=nodes, edge_index=edges, edge_attr=edge_attributes, y=y)
        # make the graph undirected
        data_graph = torch_geometric.transforms.to_undirected.ToUndirected()(data_graph)

        data, slices = self.collate([data_graph])
        torch.save((data, slices), self.processed_paths[0])

    def _load_data_and_preprocess(self):
        df_movies = pd.read_csv(os.path.join(self.raw_dir, "tmdb_5000_movies_processed.csv"))  # .iloc[0:30]
        df_credits = pd.read_csv(os.path.join(self.raw_dir, "tmdb_5000_credits_processed.csv"))  # .iloc[0:30]
        df = (
            df_movies.set_index("id")
            .join(df_credits.set_index("movie_id"), lsuffix="_movies", rsuffix="_credits")
            .reset_index()
        )

        # remove unnecessary columns
        df = df[["revenue", self.node_feature_column_source, self.edge_weight_column_source]]
        df = df.dropna()

        y = torch.from_numpy(df["revenue"].to_numpy())
        nodes = self._extract_nodes(df)
        edges, edge_attributes = self._extract_edge_weights(df)

        return nodes, edges, edge_attributes, y

    def _extract_nodes(self, df):
        transformer = self._get_node_transformer()
        node_data = self._get_data_for_node_transformer(df)
        nodes_features = transformer.fit_transform(node_data)
        nodes_features = torch.from_numpy(nodes_features.todense())
        return nodes_features

    def _get_node_transformer(self):
        if self.node_feature_method == "counter":
            transformer = CountVectorizer(**self.node_feature_params)
        elif self.node_feature_method == "tf-idf":
            transformer = TfidfVectorizer(**self.node_feature_params)
        else:
            raise Exception
        return transformer

    def _get_data_for_node_transformer(self, df):
        if self.node_feature_column_source == "overview":
            node_data = df["overview"].values.astype("U")
        elif self.node_feature_column_source == "keywords":
            node_data = df["keywords"].apply(self._extract_id)
        else:
            raise Exception
        return node_data

    @staticmethod
    def _extract_id(text):
        return " ".join([str(d["id"]) for d in json.loads(text)])

    def _extract_edge_weights(self, df):
        df_cross = df[[self.edge_weight_column_source]].reset_index()
        df_cross.columns = ["id", "people_data"]
        df_cross["people_data"] = df_cross["people_data"].apply(self._extract_id)

        # cross product
        df_cross = df_cross.merge(df_cross, how="cross")
        # remove mirror pairs
        df_cross = df_cross.loc[
            pd.DataFrame(np.sort(df_cross[["id_x", "id_y"]], 1), index=df_cross.index)
            .drop_duplicates(keep="first")
            .index
        ]

        # calculate distances
        df_cross["distance"] = df_cross.apply(
            lambda x: self.jaccard_similarity(
                np.unique(x["people_data_x"].split(" ")), np.unique(x["people_data_y"].split(" "))
            ),
            axis=1,
        )

        if self.jaccard_distance_threshold is not None:
            df_cross = df_cross[df_cross["distance"] > self.jaccard_distance_threshold]

        edges = torch.from_numpy(df_cross[["id_x", "id_y"]].to_numpy()).type(torch.long)
        edges = edges.t().contiguous()
        edge_attributes = torch.from_numpy(df_cross[["distance"]].to_numpy())

        edges, edge_attributes = torch_geometric.utils.remove_self_loops(edges, edge_attributes)
        return edges, edge_attributes

    @staticmethod
    def jaccard_similarity(x1, x2):
        intersection = len(list(set(x1).intersection(x2)))
        union = (len(x1) + len(x2)) - intersection
        return float(intersection) / union
