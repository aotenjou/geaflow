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
import java.util.Random;
import java.util.Set;

public class GCNSubgraphBuilder {

    private final GCNConfig config;

    public GCNSubgraphBuilder(GCNConfig config) {
        this.config = config;
    }

    public GCNInferPayload build(Object rootId, GraphAdapter adapter) {
        Map<Object, double[]> nodeFeatures = new LinkedHashMap<>();
        List<int[]> edges = new ArrayList<>();
        Set<Object> currentFrontier = new LinkedHashSet<>();
        currentFrontier.add(rootId);
        nodeFeatures.put(rootId, adapter.loadFeatures(rootId));
        Random random = new Random(config.getRandomSeed() ^ rootId.hashCode());
        for (int hop = 0; hop < config.getNumHops(); hop++) {
            Set<Object> nextFrontier = new LinkedHashSet<>();
            for (Object nodeId : currentFrontier) {
                List<Object> neighbors = new ArrayList<>(adapter.loadNeighbors(nodeId));
                if (neighbors.isEmpty()) {
                    continue;
                }
                Collections.shuffle(neighbors, random);
                int sampleSize = Math.min(config.getNumSamplesPerHop(), neighbors.size());
                for (int i = 0; i < sampleSize; i++) {
                    Object neighborId = neighbors.get(i);
                    nextFrontier.add(neighborId);
                    nodeFeatures.computeIfAbsent(neighborId, adapter::loadFeatures);
                    edges.add(new int[]{indexOf(nodeFeatures, nodeId), indexOf(nodeFeatures, neighborId)});
                }
            }
            currentFrontier = nextFrontier;
            if (currentFrontier.isEmpty()) {
                break;
            }
        }
        if (config.isWithSelfLoop()) {
            for (int i = 0; i < nodeFeatures.size(); i++) {
                edges.add(new int[]{i, i});
            }
        }
        List<Object> sampledNodes = new ArrayList<>(nodeFeatures.keySet());
        List<double[]> features = new ArrayList<>(nodeFeatures.values());
        int[][] edgeIndex = new int[2][edges.size()];
        for (int i = 0; i < edges.size(); i++) {
            edgeIndex[0][i] = edges.get(i)[0];
            edgeIndex[1][i] = edges.get(i)[1];
        }
        return new GCNInferPayload(rootId, sampledNodes, features, edgeIndex, null);
    }

    private int indexOf(Map<Object, double[]> nodeFeatures, Object nodeId) {
        int index = 0;
        for (Object id : nodeFeatures.keySet()) {
            if (id.equals(nodeId)) {
                return index;
            }
            index++;
        }
        throw new IllegalArgumentException("Node not found in sampled subgraph: " + nodeId);
    }

    public interface GraphAdapter extends Serializable {

        List<Object> loadNeighbors(Object nodeId);

        double[] loadFeatures(Object nodeId);
    }
}
