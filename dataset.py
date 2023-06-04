from typing import Literal, Dict

import torch
import torch_geometric
import pandas as pd
import numpy as np
import os
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


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
        add_additional_node_features: bool = True,
        edge_weight_column_source: Literal["cast", "crew", "keywords"] = "cast",
        jaccard_distance_threshold: float = 0,
        graph_type: Literal["homogenous", "heterogeneous"] = "homogenous",
    ):
        """
        The Graph Dataset representing the TMDB data.

        :param root: Root directory where the dataset should be saved.
        :param node_feature_method: Method to extract node features.
        :param node_feature_params: Dictionary with parameters for node feature extraction method.
        :param node_feature_column_source: The column from which the node features should be extracted.
        :param add_additional_node_features: If True, the additional (descriptive, not related to keywords/overview) node features are added to the dataset
        :param edge_weight_column_source: The column from which the weight features should be extracted.
        Used only if graph_type='homogenous'.
        :param jaccard_distance_threshold: The Jaccard distance threshold above which the edges are added to
        the dataset. Used only if graph_type='homogenous'.
        :param graph_type: Graph type. If "homogenous", the nodes are connected based on the jaccard distance between
        crew or cast members. If "heterogeneous", there three node types in the data: the movie nodes features are
        extracted as in "homogenous" case; the movies are connected to the nodes describing crew and cast.
        """
        self.node_feature_method = node_feature_method
        self.node_feature_column_source = node_feature_column_source
        if node_feature_params is None:
            self.node_feature_params = {}
        else:
            self.node_feature_params = node_feature_params
        self.add_additional_node_features = add_additional_node_features
        self.edge_weight_column_source = edge_weight_column_source
        self.jaccard_distance_threshold = jaccard_distance_threshold
        self.graph_type = graph_type

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["tmdb_5000_credits_processed.csv", "tmdb_5000_movies_processed.csv"]

    @property
    def processed_file_names(self):
        return (
            f"data_{self.graph_type}_{self.node_feature_method}_{self.node_feature_column_source}_"
            f"{'additional' if self.add_additional_node_features else 'noadditional'}_"
            f"{self.edge_weight_column_source}.pt"
        )

    def process(self):
        if self.graph_type == "homogenous":
            nodes, edges, edge_attributes, y = self._load_data_and_preprocess()

            #########################################################################
            # fixing the bug
            n_nodes = nodes.shape[0]
            edges = edges.numpy()
            indices = np.nonzero(np.max(edges, axis=0) <= n_nodes)[0]
            edges = edges[:, indices]
            edge_attributes = edge_attributes.numpy()[indices, :]
            edges = torch.from_numpy(edges)
            edge_attributes = torch.from_numpy(edge_attributes)
            #########################################################################

            data_graph = torch_geometric.data.Data(
                x=nodes.float(), edge_index=edges.long(), edge_attr=edge_attributes, y=y.reshape(-1, 1).float()
            )
        elif self.graph_type == "heterogeneous":
            data = self._load_and_preprocess_heterogeneous()
            data_graph = torch_geometric.data.HeteroData()

            data_graph["movies"].x = data["nodes"]["movies"]["x"]
            data_graph["crew"].x = data["nodes"]["crew"]["x"]
            data_graph["cast"].x = data["nodes"]["cast"]["x"]

            data_graph["movies"].y = data["nodes"]["movies"]["y"]

            data_graph["crew", "in", "movies"].edge_index = data["edges"]["movie_crew"][torch.tensor([1, 0]), :]
            data_graph["cast", "in", "movies"].edge_index = data["edges"]["movie_cast"][torch.tensor([1, 0]), :]
        else:
            raise Exception

        # make the graph undirected
        data_graph = torch_geometric.transforms.to_undirected.ToUndirected()(data_graph)

        try:
            print("Data was preprocessed!")
            print("Validating the graph...")
            if data_graph.validate():
                print("The graph is correct!")
        except Exception as error:
            print("The graph is not correct: ", error)

        data, slices = self.collate([data_graph])
        torch.save((data, slices), self.processed_paths[0])

    def _load_dataframe(self):
        df_movies = pd.read_csv(os.path.join(self.raw_dir, "tmdb_5000_movies_processed.csv")).drop(
            [
                "genres",
                "popularity",
                "spoken_languages",
                "production_companies",
                "production_countries",
                "vote_average",
                "vote_count",
                "title",
            ],
            axis=1,
        )  # .iloc[0:30]
        df_credits = pd.read_csv(os.path.join(self.raw_dir, "tmdb_5000_credits_processed.csv"))  # .iloc[0:30]

        df = (
            df_movies.set_index("id")
            .join(df_credits.set_index("movie_id"), lsuffix="_movies", rsuffix="_credits")
            .reset_index()
        )
        df = df.dropna()
        return df

    def _load_and_preprocess_heterogeneous(self):
        df = self._load_dataframe()
        movie_y = torch.from_numpy(df["revenue"].to_numpy())
        movie_nodes = self._extract_nodes(df)

        df = df.reset_index()
        crew_mapping, movies_crew_edges = self._extract_people_nodes(df, "crew")
        cast_mapping, movies_cast_edges = self._extract_people_nodes(df, "cast")

        movies_crew_edges = torch.from_numpy(np.array(movies_crew_edges)).type(torch.long).t().contiguous()
        movies_cast_edges = torch.from_numpy(np.array(movies_cast_edges)).type(torch.long).t().contiguous()

        return {
            "nodes": {
                "movies": {"x": movie_nodes, "y": movie_y},
                "crew": {"x": torch.eye(len(crew_mapping))},
                "cast": {"x": torch.eye(len(cast_mapping))},
            },
            "edges": {"movie_crew": movies_crew_edges, "movie_cast": movies_cast_edges},
        }

    def _extract_people_nodes(self, df, column):
        mapping = {}
        counter = 0
        edges = []

        for index, row in df.iterrows():
            movie_node_index = row["index"]
            people = self._extract_id(row[column]).split(" ")

            for person in people:
                if person not in mapping.keys():
                    mapping[person] = counter
                    counter += 1
                edges.append([movie_node_index, mapping[person]])
        return mapping, edges

    def _load_data_and_preprocess(self):
        df = self._load_dataframe()
        df = df.dropna()

        y = torch.from_numpy(df["revenue"].to_numpy())
        nodes = self._extract_nodes(df)

        edges, edge_attributes = self._extract_edge_weights(df)
        return nodes, edges, edge_attributes, y

    def _extract_nodes(self, df):
        transformer = self._get_node_transformer()
        node_data = self._get_data_for_node_transformer(df)
        nodes_features = transformer.fit_transform(node_data).todense()
        print(f"Added {nodes_features.shape[1]} node features")
        if self.add_additional_node_features:
            nodes_features = self._add_additional_node_features(df, nodes_features)
        nodes_features = torch.from_numpy(nodes_features)
        return nodes_features

    def _add_additional_node_features(self, df, nodes_features):
        additional_features = df.drop(["revenue", "keywords", "overview", "title", "cast", "crew", "id"], axis=1)
        additional_features[["budget", "runtime", "release_year", "release_month"]] = StandardScaler().fit_transform(
            additional_features[["budget", "runtime", "release_year", "release_month"]]
        )
        print(f"Added {additional_features.shape[1]} additional node features")
        nodes_features = np.hstack([additional_features.values, nodes_features])
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
