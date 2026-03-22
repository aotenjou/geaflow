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
import java.util.List;

public class GCNInferPayload implements Serializable {

    private final Object centerNodeId;
    private final List<Object> sampledNodes;
    private final List<double[]> nodeFeatures;
    private final int[][] edgeIndex;
    private final double[] edgeWeight;

    public GCNInferPayload(Object centerNodeId, List<Object> sampledNodes, List<double[]> nodeFeatures,
                           int[][] edgeIndex, double[] edgeWeight) {
        this.centerNodeId = centerNodeId;
        this.sampledNodes = sampledNodes;
        this.nodeFeatures = nodeFeatures;
        this.edgeIndex = edgeIndex;
        this.edgeWeight = edgeWeight;
    }

    public Object getCenter_node_id() {
        return centerNodeId;
    }

    public List<Object> getSampled_nodes() {
        return sampledNodes;
    }

    public List<double[]> getNode_features() {
        return nodeFeatures;
    }

    public int[][] getEdge_index() {
        return edgeIndex;
    }

    public double[] getEdge_weight() {
        return edgeWeight;
    }
}
