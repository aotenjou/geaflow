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
import org.apache.geaflow.api.graph.compute.IncVertexCentricCompute;
import org.apache.geaflow.api.graph.function.vc.IncVertexCentricComputeFunction;
import org.apache.geaflow.api.graph.function.vc.VertexCentricCombineFunction;
import org.apache.geaflow.api.graph.function.vc.base.IncGraphInferContext;
import org.apache.geaflow.dsl.udf.graph.gcn.AbstractGCNSubgraphAdapter;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNConfig;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNFeatureCollector;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferPayload;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferResult;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNResultParser;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNSubgraphBuilder;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.vertex.IVertex;
import org.apache.geaflow.model.graph.vertex.impl.ValueVertex;

public class GCNCompute extends IncVertexCentricCompute<Object, List<Object>, Object, Object> {

    private final GCNConfig config;

    public GCNCompute() {
        this(GCNConfig.DEFAULT_NUM_HOPS, GCNConfig.DEFAULT_NUM_SAMPLES_PER_HOP,
            GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS);
    }

    public GCNCompute(int numHops, int numSamplesPerHop) {
        this(numHops, numSamplesPerHop, GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS);
    }

    public GCNCompute(int numHops, int numSamplesPerHop, String pythonTransformClassName) {
        super(1L);
        this.config = new GCNConfig(numHops, numSamplesPerHop, pythonTransformClassName);
    }

    @Override
    public String getPythonTransformClassName() {
        return config.getPythonTransformClassName();
    }

    @Override
    public IncVertexCentricComputeFunction<Object, List<Object>, Object, Object> getIncComputeFunction() {
        return new GCNComputeFunction(config);
    }

    @Override
    public VertexCentricCombineFunction<Object> getCombineFunction() {
        return null;
    }

    static class GCNComputeFunction implements IncVertexCentricComputeFunction<Object, List<Object>, Object, Object> {

        private final GCNConfig config;
        private final GCNFeatureCollector featureCollector = new GCNFeatureCollector();
        private final GCNResultParser resultParser = new GCNResultParser();
        private final GCNSubgraphBuilder subgraphBuilder;
        private IncGraphComputeContext<Object, List<Object>, Object, Object> graphContext;
        private IncGraphInferContext<Object> inferContext;

        GCNComputeFunction(GCNConfig config) {
            this.config = config;
            this.subgraphBuilder = new GCNSubgraphBuilder(config);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void init(IncGraphComputeContext<Object, List<Object>, Object, Object> incGraphContext) {
            this.graphContext = incGraphContext;
            if (incGraphContext instanceof IncGraphInferContext) {
                this.inferContext = (IncGraphInferContext<Object>) incGraphContext;
            }
        }

        @Override
        public void evolve(Object vertexId, TemporaryGraph<Object, List<Object>, Object> temporaryGraph) {
            inferAndCollect(vertexId, temporaryGraph.getVertex());
        }

        @Override
        public void compute(Object vertexId, Iterator<Object> messageIterator) {
        }

        @Override
        public void finish(Object vertexId, MutableGraph<Object, List<Object>, Object> mutableGraph) {
        }

        private void inferAndCollect(Object rootId, IVertex<Object, List<Object>> rootVertex) {
            if (inferContext == null) {
                throw new IllegalStateException("GCNCompute requires infer-enabled runtime context");
            }
            GCNInferPayload payload = subgraphBuilder.build(rootId, new ApiGraphAdapter(rootId, rootVertex));
            Object rawResult = inferContext.infer(payload);
            GCNInferResult result = resultParser.parse(rootId, rawResult);
            List<Object> value = new ArrayList<>();
            value.add(result.getEmbedding());
            value.add(result.getPredictedClass());
            value.add(result.getConfidence());
            graphContext.collect(new ValueVertex<>(rootId, value));
        }

        private class ApiGraphAdapter extends AbstractGCNSubgraphAdapter<IVertex<Object, List<Object>>> {

            ApiGraphAdapter(Object rootId, IVertex<Object, List<Object>> rootVertex) {
                super(rootId, rootVertex, config.getFeatureDimLimit());
            }

            @Override
            protected List<? extends IEdge<Object, ?>> loadEdges(Object nodeId) {
                Object query = graphContext.getHistoricalGraph()
                    .getSnapShot(graphContext.getHistoricalGraph().getLatestVersionId()).edges();
                try {
                    Method withId = query.getClass().getMethod("withId", Object.class);
                    withId.invoke(query, nodeId);
                    Method getEdges = query.getClass().getMethod("getEdges");
                    return (List<IEdge<Object, Object>>) getEdges.invoke(query);
                } catch (NoSuchMethodException e) {
                    return Collections.emptyList();
                } catch (IllegalAccessException | InvocationTargetException e) {
                    throw new IllegalStateException("Failed to load neighbors for GCNCompute", e);
                }
            }

            @Override
            protected IVertex<Object, List<Object>> loadNode(Object nodeId) {
                return graphContext.getHistoricalGraph().getSnapShot(
                    graphContext.getHistoricalGraph().getLatestVersionId())
                    .vertex().withId(nodeId).get();
            }

            @Override
            protected double[] extractFeatures(IVertex<Object, List<Object>> vertex) {
                return featureCollector.collectFromValue(vertex.getValue(), config.getFeatureDimLimit());
            }
        }
    }
}
