# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import abc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class TransFormFunction(abc.ABC):

    def __init__(self, input_size):
        self.input_size = input_size

    @abc.abstractmethod
    def load_model(self, *args):
        pass

    @abc.abstractmethod
    def transform_pre(self, *args):
        pass

    @abc.abstractmethod
    def transform_post(self, *args):
        pass


class EmptyGCNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, num_classes):
        super(EmptyGCNModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, node_features, edge_index, edge_weight=None):
        hidden = self._aggregate(node_features, edge_index, edge_weight)
        hidden = F.relu(self.input_proj(hidden))
        embedding = self.embedding_proj(self._aggregate(hidden, edge_index, edge_weight))
        logits = self.classifier(embedding)
        return embedding, logits

    def _aggregate(self, features, edge_index, edge_weight=None):
        if edge_index is None or edge_index.numel() == 0:
            return features
        num_nodes = features.size(0)
        aggregated = torch.zeros_like(features)
        degree = torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device)
        src_index = edge_index[0]
        dst_index = edge_index[1]
        for idx in range(src_index.size(0)):
            src = int(src_index[idx].item())
            dst = int(dst_index[idx].item())
            if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
                continue
            weight = 1.0
            if edge_weight is not None and idx < edge_weight.size(0):
                weight = float(edge_weight[idx].item())
            aggregated[dst] = aggregated[dst] + weight * features[src]
            degree[dst] = degree[dst] + weight
        degree = torch.clamp(degree, min=1.0)
        return aggregated / degree


class GCNTransFormFunction(TransFormFunction):

    def __init__(self):
        super(GCNTransFormFunction, self).__init__(input_size=1)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.hidden_dim = 32
        self.embedding_dim = 16
        self.num_classes = 2
        self.model = None
        self.model_path = os.path.join(os.getcwd(), "model.pt")
        self.model_loaded = False
        self.load_model(self.model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        if not os.path.exists(model_path):
            self.model = None
            self.model_loaded = False
            return
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint.to(self.device)
            self.model.eval()
            self.model_loaded = True
            return
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported model checkpoint type: {}".format(type(checkpoint)))
        state_dict = checkpoint.get("state_dict", checkpoint)
        input_dim = int(checkpoint.get("input_dim", state_dict["input_proj.weight"].shape[1]))
        hidden_dim = int(checkpoint.get("hidden_dim", state_dict["input_proj.weight"].shape[0]))
        embedding_dim = int(checkpoint.get("embedding_dim", state_dict["embedding_proj.weight"].shape[0]))
        num_classes = int(checkpoint.get("num_classes", state_dict["classifier.weight"].shape[0]))
        self.model = EmptyGCNModel(input_dim, hidden_dim, embedding_dim, num_classes).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_loaded = True

    def transform_pre(self, *args):
        payload = args[0]
        center_node_id = self._read_value(payload, ["center_node_id", "centerNodeId"],
                                          ["getCenter_node_id", "getCenterNodeId"])
        sampled_nodes = self._read_value(payload, ["sampled_nodes", "sampledNodes"],
                                         ["getSampled_nodes", "getSampledNodes"])
        node_features = self._read_value(payload, ["node_features", "nodeFeatures"],
                                         ["getNode_features", "getNodeFeatures"])
        edge_index = self._read_value(payload, ["edge_index", "edgeIndex"],
                                      ["getEdge_index", "getEdgeIndex"])
        edge_weight = self._read_value(payload, ["edge_weight", "edgeWeight"],
                                       ["getEdge_weight", "getEdgeWeight"])

        feature_tensor = self._build_feature_tensor(node_features)
        edge_tensor = self._build_edge_tensor(edge_index, feature_tensor.size(0))
        edge_weight_tensor = self._build_edge_weight_tensor(edge_weight, edge_tensor.size(1))
        self._ensure_model(feature_tensor.size(1))

        with torch.no_grad():
            embedding_matrix, logits = self.model(feature_tensor, edge_tensor, edge_weight_tensor)

        center_index = self._find_center_index(sampled_nodes, center_node_id)
        center_embedding = embedding_matrix[center_index]
        center_logits = logits[center_index]
        probs = F.softmax(center_logits, dim=0)
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())

        result = {
            "node_id": center_node_id,
            "embedding": center_embedding.cpu().tolist(),
            "predicted_class": predicted_class,
            "confidence": confidence
        }
        return result, center_node_id

    def transform_post(self, *args):
        if len(args) == 0:
            return None
        return args[0]

    def _ensure_model(self, input_dim):
        if self.model is not None and self.model.input_proj.in_features == input_dim:
            return
        torch.manual_seed(20260322)
        self.model = EmptyGCNModel(input_dim, self.hidden_dim, self.embedding_dim,
                                   self.num_classes).to(self.device)
        self.model.eval()

    def _build_feature_tensor(self, node_features):
        if node_features is None or len(node_features) == 0:
            return torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        rows = []
        for feature_row in node_features:
            if feature_row is None:
                rows.append([0.0])
                continue
            row = []
            for value in feature_row:
                row.append(float(0.0 if value is None else value))
            if len(row) == 0:
                row.append(0.0)
            rows.append(row)
        max_dim = 1
        for row in rows:
            if len(row) > max_dim:
                max_dim = len(row)
        normalized_rows = []
        for row in rows:
            if len(row) < max_dim:
                row = row + [0.0] * (max_dim - len(row))
            normalized_rows.append(row)
        return torch.tensor(normalized_rows, dtype=torch.float32, device=self.device)

    def _build_edge_tensor(self, edge_index, num_nodes):
        if edge_index is None:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        if len(edge_index) < 2:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        src_list = list(edge_index[0])
        dst_list = list(edge_index[1])
        edges = []
        for idx in range(min(len(src_list), len(dst_list))):
            src = int(src_list[idx])
            dst = int(dst_list[idx])
            if src < 0 or dst < 0:
                continue
            if src >= num_nodes or dst >= num_nodes:
                continue
            edges.append([src, dst])
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        edge_tensor = torch.tensor(edges, dtype=torch.long, device=self.device)
        return edge_tensor.t().contiguous()

    def _build_edge_weight_tensor(self, edge_weight, num_edges):
        if num_edges == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        if edge_weight is None:
            return None
        values = list(edge_weight)
        normalized_weights = []
        for idx in range(num_edges):
            value = values[idx] if idx < len(values) else 1.0
            normalized_weights.append(float(1.0 if value is None else value))
        return torch.tensor(normalized_weights, dtype=torch.float32, device=self.device)

    def _find_center_index(self, sampled_nodes, center_node_id):
        if sampled_nodes is None or len(sampled_nodes) == 0:
            return 0
        for idx in range(len(sampled_nodes)):
            if sampled_nodes[idx] == center_node_id:
                return idx
        return 0

    def _read_value(self, obj, attr_names, method_names):
        if isinstance(obj, dict):
            for name in attr_names:
                if name in obj:
                    return obj[name]
        for name in attr_names:
            if hasattr(obj, name):
                return getattr(obj, name)
        for name in method_names:
            if hasattr(obj, name):
                method = getattr(obj, name)
                if callable(method):
                    return method()
        return None
