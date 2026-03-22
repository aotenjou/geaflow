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
import java.util.Arrays;

public class GCNInferResult implements Serializable {

    private final Object nodeId;
    private final double[] embedding;
    private final long predictedClass;
    private final double confidence;

    public GCNInferResult(Object nodeId, double[] embedding, long predictedClass, double confidence) {
        this.nodeId = nodeId;
        this.embedding = embedding;
        this.predictedClass = predictedClass;
        this.confidence = confidence;
    }

    public Object getNodeId() {
        return nodeId;
    }

    public double[] getEmbedding() {
        return embedding;
    }

    public long getPredictedClass() {
        return predictedClass;
    }

    public double getConfidence() {
        return confidence;
    }

    public Object[] toRowValues() {
        Double[] boxedEmbedding = Arrays.stream(embedding).boxed().toArray(Double[]::new);
        return new Object[]{nodeId, boxedEmbedding, predictedClass, confidence};
    }
}
