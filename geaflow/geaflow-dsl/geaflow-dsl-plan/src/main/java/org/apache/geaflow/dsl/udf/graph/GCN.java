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

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.apache.geaflow.common.config.ConfigHelper;
import org.apache.geaflow.common.config.Configuration;
import org.apache.geaflow.common.config.InferConfigHelper;
import org.apache.geaflow.common.config.keys.FrameworkConfigKeys;
import org.apache.geaflow.common.exception.GeaflowRuntimeException;
import org.apache.geaflow.common.type.Types;
import org.apache.geaflow.dsl.common.algo.AlgorithmRuntimeContext;
import org.apache.geaflow.dsl.common.algo.AlgorithmUserFunction;
import org.apache.geaflow.dsl.common.algo.DynamicVertexRuntimeContext;
import org.apache.geaflow.dsl.common.algo.IncrementalAlgorithmUserFunction;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.data.RowVertex;
import org.apache.geaflow.dsl.common.data.impl.ObjectRow;
import org.apache.geaflow.dsl.common.function.Description;
import org.apache.geaflow.dsl.common.types.ArrayType;
import org.apache.geaflow.dsl.common.types.GraphSchema;
import org.apache.geaflow.dsl.common.types.StructType;
import org.apache.geaflow.dsl.common.types.TableField;
import org.apache.geaflow.dsl.udf.graph.gcn.AbstractGCNSubgraphAdapter;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNConfig;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNFeatureCollector;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferPayload;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferResult;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNResultParser;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNSubgraphBuilder;
import org.apache.geaflow.infer.InferContext;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;

@Description(name = "gcn", description = "built-in udga for GCN inference")
public class GCN implements AlgorithmUserFunction<Object, Object>, IncrementalAlgorithmUserFunction {

    private AlgorithmRuntimeContext<Object, Object> context;
    private DynamicVertexRuntimeContext<Object, Object> dynamicContext;
    private GCNConfig config;
    private GCNSubgraphBuilder subgraphBuilder;
    private InferContext<Object> localInferContext;
    private final GCNFeatureCollector featureCollector = new GCNFeatureCollector();
    private final GCNResultParser resultParser = new GCNResultParser();

    @Override
    public void init(AlgorithmRuntimeContext<Object, Object> context, Object[] params) {
        this.context = context;
        this.dynamicContext = context instanceof DynamicVertexRuntimeContext
            ? (DynamicVertexRuntimeContext<Object, Object>) context : null;
        this.config = parseConfig(params);
        this.subgraphBuilder = new GCNSubgraphBuilder(config);
        this.localInferContext = createLocalInferContext(context.getConfig());
    }

    @Override
    public void process(RowVertex vertex, Optional<Row> updatedValues, Iterator<Object> messages) {
        if (context.getCurrentIterationId() > 1) {
            return;
        }
        GCNInferPayload payload = subgraphBuilder.build(vertex.getId(), new DynamicGraphAdapter(vertex));
        Object rawResult = infer(payload);
        GCNInferResult result = resultParser.parse(vertex.getId(), rawResult);
        context.take(ObjectRow.create(result.toRowValues()));
    }

    @Override
    public void finish(RowVertex graphVertex, Optional<Row> updatedValues) {
        if (localInferContext != null) {
            localInferContext.close();
            localInferContext = null;
        }
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

    private Object infer(GCNInferPayload payload) {
        if (localInferContext == null) {
            return context.infer(payload);
        }
        try {
            return localInferContext.infer(payload);
        } catch (Exception e) {
            throw new GeaflowRuntimeException("GCN local infer failed", e);
        }
    }

    private InferContext<Object> createLocalInferContext(Configuration baseConfig) {
        if (!ConfigHelper.getBooleanOrDefault(baseConfig.getConfigMap(),
            FrameworkConfigKeys.INFER_ENV_ENABLE.getKey(), false)) {
            return null;
        }
        String globalClassName = InferConfigHelper.getTransformClassName(baseConfig);
        if (Objects.equals(config.getPythonTransformClassName(), globalClassName)) {
            return null;
        }
        return createInferContext(InferConfigHelper.buildInferConfiguration(baseConfig,
            config.getPythonTransformClassName()));
    }

    protected InferContext<Object> createInferContext(Configuration configuration) {
        return new InferContext<>(configuration);
    }

    private class DynamicGraphAdapter extends AbstractGCNSubgraphAdapter<RowVertex> {

        private final GraphSchema graphSchema = context.getGraphSchema();
        private final Object rootId;

        DynamicGraphAdapter(RowVertex rootVertex) {
            super(rootVertex.getId(), rootVertex, config.getFeatureDimLimit());
            this.rootId = rootVertex.getId();
        }

        @Override
        protected List<? extends IEdge<Object, ?>> loadEdges(Object nodeId) {
            RowVertex current = switchVertex(nodeId);
            try {
                if (current == null) {
                    return Collections.emptyList();
                }
                return context.loadEdges(EdgeDirection.BOTH);
            } finally {
                restoreVertex();
            }
        }

        @Override
        protected RowVertex loadNode(Object nodeId) {
            RowVertex vertex = switchVertex(nodeId);
            try {
                return vertex;
            } finally {
                restoreVertex();
            }
        }

        @Override
        protected double[] extractFeatures(RowVertex vertex) {
            return featureCollector.collectFromRowVertex(vertex, graphSchema,
                config.getFeatureDimLimit());
        }

        private RowVertex switchVertex(Object nodeId) {
            if (dynamicContext == null) {
                throw new IllegalStateException(
                    "GCN requires DynamicVertexRuntimeContext to load sampled vertices");
            }
            dynamicContext.setVertexId(nodeId);
            return dynamicContext.loadVertex();
        }

        private void restoreVertex() {
            if (dynamicContext == null) {
                throw new IllegalStateException(
                    "GCN requires DynamicVertexRuntimeContext to restore sampled vertices");
            }
            dynamicContext.setVertexId(rootId);
        }
    }
}
