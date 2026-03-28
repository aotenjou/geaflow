/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.geaflow.dsl.udf.graph.gcn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.geaflow.model.graph.edge.IEdge;

public abstract class AbstractGCNSubgraphAdapter<N> implements GCNSubgraphBuilder.GraphAdapter {

    private final Object rootId;
    private final N rootNode;
    private final int featureDimLimit;

    protected AbstractGCNSubgraphAdapter(Object rootId, N rootNode, int featureDimLimit) {
        this.rootId = rootId;
        this.rootNode = rootNode;
        this.featureDimLimit = featureDimLimit;
    }

    @Override
    public final List<GCNSubgraphBuilder.Neighbor> loadWeightedNeighbors(Object nodeId) {
        List<? extends IEdge<Object, ?>> edges = loadEdges(nodeId);
        if (edges == null || edges.isEmpty()) {
            return Collections.emptyList();
        }
        List<GCNSubgraphBuilder.Neighbor> neighbors = new ArrayList<>(edges.size());
        for (IEdge<Object, ?> edge : edges) {
            Object neighborId = nodeId.equals(edge.getSrcId()) ? edge.getTargetId() : edge.getSrcId();
            neighbors.add(new GCNSubgraphBuilder.Neighbor(neighborId,
                GCNEdgeWeightExtractor.extract(edge.getValue())));
        }
        return neighbors;
    }

    @Override
    public final double[] loadFeatures(Object nodeId) {
        N node = rootId.equals(nodeId) ? rootNode : loadNode(nodeId);
        return node == null ? new double[featureDimLimit] : extractFeatures(node);
    }

    protected abstract List<? extends IEdge<Object, ?>> loadEdges(Object nodeId);

    protected abstract N loadNode(Object nodeId);

    protected abstract double[] extractFeatures(N node);
}
