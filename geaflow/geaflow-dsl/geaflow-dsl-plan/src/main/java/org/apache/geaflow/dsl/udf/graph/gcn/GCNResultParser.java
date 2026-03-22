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

import java.lang.reflect.Array;
import java.util.List;
import java.util.Map;

public class GCNResultParser {

    public GCNInferResult parse(Object expectedNodeId, Object rawResult) {
        if (rawResult == null) {
            return new GCNInferResult(expectedNodeId, new double[0], -1L, 0D);
        }
        if (rawResult instanceof String && ((String) rawResult).startsWith("python_exception:")) {
            throw new IllegalStateException((String) rawResult);
        }
        if (rawResult instanceof GCNInferResult) {
            return (GCNInferResult) rawResult;
        }
        if (rawResult instanceof Map) {
            return parseMap(expectedNodeId, (Map<?, ?>) rawResult);
        }
        if (rawResult instanceof List) {
            return parseList(expectedNodeId, (List<?>) rawResult);
        }
        if (rawResult.getClass().isArray()) {
            return parseArray(expectedNodeId, rawResult);
        }
        throw new IllegalArgumentException("Unsupported GCN infer result type: "
            + rawResult.getClass().getName());
    }

    private GCNInferResult parseMap(Object expectedNodeId, Map<?, ?> rawResult) {
        Object nodeId = rawResult.containsKey("node_id") ? rawResult.get("node_id")
            : (rawResult.containsKey("id") ? rawResult.get("id") : expectedNodeId);
        Object embedding = rawResult.get("embedding");
        Object prediction = rawResult.containsKey("prediction") ? rawResult.get("prediction")
            : rawResult.get("predicted_class");
        Object confidence = rawResult.get("confidence");
        return new GCNInferResult(nodeId, parseEmbedding(embedding), parseLong(prediction),
            parseDouble(confidence));
    }

    private GCNInferResult parseList(Object expectedNodeId, List<?> rawResult) {
        if (rawResult.size() >= 4) {
            return new GCNInferResult(rawResult.get(0), parseEmbedding(rawResult.get(1)),
                parseLong(rawResult.get(2)), parseDouble(rawResult.get(3)));
        }
        if (rawResult.size() >= 3) {
            return new GCNInferResult(expectedNodeId, parseEmbedding(rawResult.get(0)),
                parseLong(rawResult.get(1)), parseDouble(rawResult.get(2)));
        }
        if (rawResult.size() == 1) {
            return new GCNInferResult(expectedNodeId, parseEmbedding(rawResult.get(0)), -1L, 0D);
        }
        throw new IllegalArgumentException("GCN infer result list is empty");
    }

    private GCNInferResult parseArray(Object expectedNodeId, Object rawResult) {
        int length = Array.getLength(rawResult);
        if (length >= 4) {
            return new GCNInferResult(Array.get(rawResult, 0), parseEmbedding(Array.get(rawResult, 1)),
                parseLong(Array.get(rawResult, 2)), parseDouble(Array.get(rawResult, 3)));
        }
        if (length >= 3) {
            return new GCNInferResult(expectedNodeId, parseEmbedding(Array.get(rawResult, 0)),
                parseLong(Array.get(rawResult, 1)), parseDouble(Array.get(rawResult, 2)));
        }
        if (length == 1) {
            return new GCNInferResult(expectedNodeId, parseEmbedding(Array.get(rawResult, 0)), -1L, 0D);
        }
        throw new IllegalArgumentException("GCN infer result array is empty");
    }

    private double[] parseEmbedding(Object embedding) {
        if (embedding == null) {
            return new double[0];
        }
        if (embedding instanceof double[]) {
            return (double[]) embedding;
        }
        if (embedding instanceof Double[]) {
            Double[] boxed = (Double[]) embedding;
            double[] values = new double[boxed.length];
            for (int i = 0; i < boxed.length; i++) {
                values[i] = boxed[i] == null ? 0D : boxed[i];
            }
            return values;
        }
        if (embedding instanceof List) {
            List<?> list = (List<?>) embedding;
            double[] values = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                values[i] = parseDouble(list.get(i));
            }
            return values;
        }
        if (embedding.getClass().isArray()) {
            int len = Array.getLength(embedding);
            double[] values = new double[len];
            for (int i = 0; i < len; i++) {
                values[i] = parseDouble(Array.get(embedding, i));
            }
            return values;
        }
        return new double[]{parseDouble(embedding)};
    }

    private long parseLong(Object value) {
        if (value == null) {
            return -1L;
        }
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        return Long.parseLong(String.valueOf(value));
    }

    private double parseDouble(Object value) {
        if (value == null) {
            return 0D;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return Double.parseDouble(String.valueOf(value));
    }
}
