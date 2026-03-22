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

package org.apache.geaflow.dsl.udf.graph;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import org.apache.geaflow.common.type.Types;
import org.apache.geaflow.dsl.common.algo.AlgorithmRuntimeContext;
import org.apache.geaflow.dsl.common.algo.AlgorithmUserFunction;
import org.apache.geaflow.dsl.common.algo.IncrementalAlgorithmUserFunction;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.data.RowEdge;
import org.apache.geaflow.dsl.common.data.RowVertex;
import org.apache.geaflow.dsl.common.data.impl.ObjectRow;
import org.apache.geaflow.dsl.common.function.Description;
import org.apache.geaflow.dsl.common.types.ArrayType;
import org.apache.geaflow.dsl.common.types.GraphSchema;
import org.apache.geaflow.dsl.common.types.StructType;
import org.apache.geaflow.dsl.common.types.TableField;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNConfig;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNFeatureCollector;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferPayload;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferResult;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNResultParser;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNSubgraphBuilder;
import org.apache.geaflow.model.graph.edge.EdgeDirection;

@Description(name = "gcn", description = "built-in udga for GCN inference")
public class GCN implements AlgorithmUserFunction<Object, Object>, IncrementalAlgorithmUserFunction {

    private AlgorithmRuntimeContext<Object, Object> context;
    private GCNConfig config;
    private GCNSubgraphBuilder subgraphBuilder;
    private final GCNFeatureCollector featureCollector = new GCNFeatureCollector();
    private final GCNResultParser resultParser = new GCNResultParser();

    @Override
    public void init(AlgorithmRuntimeContext<Object, Object> context, Object[] params) {
        this.context = context;
        this.config = parseConfig(params);
        this.subgraphBuilder = new GCNSubgraphBuilder(config);
    }

    @Override
    public void process(RowVertex vertex, Optional<Row> updatedValues, Iterator<Object> messages) {
        if (context.getCurrentIterationId() > 1) {
            return;
        }
        GCNInferPayload payload = subgraphBuilder.build(vertex.getId(), new DynamicGraphAdapter(vertex));
        Object rawResult = context.infer(payload);
        GCNInferResult result = resultParser.parse(vertex.getId(), rawResult);
        context.take(ObjectRow.create(result.toRowValues()));
    }

    @Override
    public void finish(RowVertex graphVertex, Optional<Row> updatedValues) {
    }

    @Override
    public StructType getOutputType(GraphSchema graphSchema) {
        return new StructType(
            new TableField("id", graphSchema.getIdType(), false),
            new TableField("embedding", new ArrayType(Types.DOUBLE), false),
            new TableField("predicted_class", Types.LONG, false),
            new TableField("confidence", Types.DOUBLE, false)
        );
    }

    private GCNConfig parseConfig(Object[] params) {
        if (params.length == 0) {
            return new GCNConfig();
        }
        if (params.length != 2 && params.length != 3) {
            throw new IllegalArgumentException("GCN accepts 0, 2 or 3 parameters");
        }
        int numHops = Integer.parseInt(String.valueOf(params[0]));
        int numSamplesPerHop = Integer.parseInt(String.valueOf(params[1]));
        String className = params.length == 3 ? String.valueOf(params[2])
            : GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS;
        return new GCNConfig(numHops, numSamplesPerHop, className);
    }

    private class DynamicGraphAdapter implements GCNSubgraphBuilder.GraphAdapter {

        private final GraphSchema graphSchema = context.getGraphSchema();
        private final RowVertex rootVertex;

        DynamicGraphAdapter(RowVertex rootVertex) {
            this.rootVertex = rootVertex;
        }

        @Override
        public List<Object> loadNeighbors(Object nodeId) {
            RowVertex current = switchVertex(nodeId);
            if (current == null) {
                restoreVertex(rootVertex.getId());
                return Collections.emptyList();
            }
            List<RowEdge> edges = context.loadEdges(EdgeDirection.BOTH);
            List<Object> neighbors = new ArrayList<>(edges.size());
            for (RowEdge edge : edges) {
                Object neighborId = nodeId.equals(edge.getSrcId()) ? edge.getTargetId()
                    : edge.getSrcId();
                neighbors.add(neighborId);
            }
            restoreVertex(rootVertex.getId());
            return neighbors;
        }

        @Override
        public double[] loadFeatures(Object nodeId) {
            RowVertex vertex = nodeId.equals(rootVertex.getId()) ? rootVertex : switchVertex(nodeId);
            try {
                return vertex == null ? new double[config.getFeatureDimLimit()]
                    : featureCollector.collectFromRowVertex(vertex, graphSchema,
                        config.getFeatureDimLimit());
            } finally {
                restoreVertex(rootVertex.getId());
            }
        }

        private RowVertex switchVertex(Object nodeId) {
            try {
                Method setVertexId = context.getClass().getMethod("setVertexId", Object.class);
                setVertexId.invoke(context, nodeId);
                Method loadVertex = context.getClass().getMethod("loadVertex");
                return (RowVertex) loadVertex.invoke(context);
            } catch (NoSuchMethodException e) {
                throw new IllegalStateException("GCN requires dynamic runtime context with loadVertex support", e);
            } catch (IllegalAccessException | InvocationTargetException e) {
                throw new IllegalStateException("Failed to switch runtime context vertex", e);
            }
        }

        private void restoreVertex(Object rootId) {
            try {
                Method setVertexId = context.getClass().getMethod("setVertexId", Object.class);
                setVertexId.invoke(context, rootId);
            } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
                throw new IllegalStateException("Failed to restore runtime context vertex", e);
            }
        }
    }
}
