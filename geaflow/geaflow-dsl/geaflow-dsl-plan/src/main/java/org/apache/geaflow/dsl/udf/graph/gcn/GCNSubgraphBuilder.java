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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

public class GCNSubgraphBuilder {

    private final GCNConfig config;

    public GCNSubgraphBuilder(GCNConfig config) {
        this.config = config;
    }

    public GCNInferPayload build(Object rootId, GraphAdapter adapter) {
        Map<Object, double[]> nodeFeatures = new LinkedHashMap<>();
        Map<Object, Integer> nodeIndexes = new LinkedHashMap<>();
        Map<DirectedEdge, Double> edgeWeights = new LinkedHashMap<>();
        Set<Object> currentFrontier = new LinkedHashSet<>();
        currentFrontier.add(rootId);
        putNode(nodeFeatures, nodeIndexes, rootId, adapter.loadFeatures(rootId));
        Random random = new Random(config.getRandomSeed() ^ rootId.hashCode());
        for (int hop = 0; hop < config.getNumHops(); hop++) {
            Set<Object> nextFrontier = new LinkedHashSet<>();
            for (Object nodeId : currentFrontier) {
                List<Neighbor> neighbors = new ArrayList<>(adapter.loadWeightedNeighbors(nodeId));
                if (neighbors.isEmpty()) {
                    continue;
                }
                Collections.shuffle(neighbors, random);
                int sampleSize = Math.min(config.getNumSamplesPerHop(), neighbors.size());
                int srcIndex = nodeIndexes.get(nodeId);
                for (int i = 0; i < sampleSize; i++) {
                    Neighbor neighbor = neighbors.get(i);
                    Object neighborId = neighbor.getNodeId();
                    nextFrontier.add(neighborId);
                    if (!nodeFeatures.containsKey(neighborId)) {
                        putNode(nodeFeatures, nodeIndexes, neighborId, adapter.loadFeatures(neighborId));
                    }
                    int dstIndex = nodeIndexes.get(neighborId);
                    putEdge(edgeWeights, srcIndex, dstIndex, neighbor.getWeight());
                    putEdge(edgeWeights, dstIndex, srcIndex, neighbor.getWeight());
                }
            }
            currentFrontier = nextFrontier;
            if (currentFrontier.isEmpty()) {
                break;
            }
        }
        if (config.isWithSelfLoop()) {
            for (int i = 0; i < nodeFeatures.size(); i++) {
                putEdge(edgeWeights, i, i, GCNEdgeWeightExtractor.DEFAULT_EDGE_WEIGHT);
            }
        }
        List<Object> sampledNodes = new ArrayList<>(nodeFeatures.keySet());
        List<double[]> features = new ArrayList<>(nodeFeatures.values());
        int[][] edgeIndex = new int[2][edgeWeights.size()];
        double[] payloadEdgeWeight = new double[edgeWeights.size()];
        int index = 0;
        for (Map.Entry<DirectedEdge, Double> entry : edgeWeights.entrySet()) {
            edgeIndex[0][index] = entry.getKey().getSrcIndex();
            edgeIndex[1][index] = entry.getKey().getDstIndex();
            payloadEdgeWeight[index] = entry.getValue();
            index++;
        }
        return new GCNInferPayload(rootId, sampledNodes, features, edgeIndex, payloadEdgeWeight);
    }

    private void putNode(Map<Object, double[]> nodeFeatures, Map<Object, Integer> nodeIndexes,
                         Object nodeId, double[] features) {
        nodeIndexes.put(nodeId, nodeIndexes.size());
        nodeFeatures.put(nodeId, features);
    }

    private void putEdge(Map<DirectedEdge, Double> edgeWeights, int srcIndex, int dstIndex,
                         double weight) {
        edgeWeights.putIfAbsent(new DirectedEdge(srcIndex, dstIndex), weight);
    }

    public interface GraphAdapter extends Serializable {

        default List<Object> loadNeighbors(Object nodeId) {
            List<Neighbor> neighbors = loadWeightedNeighbors(nodeId);
            if (neighbors == null || neighbors.isEmpty()) {
                return Collections.emptyList();
            }
            List<Object> neighborIds = new ArrayList<>(neighbors.size());
            for (Neighbor neighbor : neighbors) {
                neighborIds.add(neighbor.getNodeId());
            }
            return neighborIds;
        }

        List<Neighbor> loadWeightedNeighbors(Object nodeId);

        double[] loadFeatures(Object nodeId);
    }

    public static final class Neighbor implements Serializable {

        private final Object nodeId;
        private final double weight;

        public Neighbor(Object nodeId, double weight) {
            this.nodeId = nodeId;
            this.weight = weight;
        }

        public Object getNodeId() {
            return nodeId;
        }

        public double getWeight() {
            return weight;
        }
    }

    private static final class DirectedEdge implements Serializable {

        private final int srcIndex;
        private final int dstIndex;

        private DirectedEdge(int srcIndex, int dstIndex) {
            this.srcIndex = srcIndex;
            this.dstIndex = dstIndex;
        }

        private int getSrcIndex() {
            return srcIndex;
        }

        private int getDstIndex() {
            return dstIndex;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof DirectedEdge)) {
                return false;
            }
            DirectedEdge that = (DirectedEdge) o;
            return srcIndex == that.srcIndex && dstIndex == that.dstIndex;
        }

        @Override
        public int hashCode() {
            return Objects.hash(srcIndex, dstIndex);
        }
    }
}
