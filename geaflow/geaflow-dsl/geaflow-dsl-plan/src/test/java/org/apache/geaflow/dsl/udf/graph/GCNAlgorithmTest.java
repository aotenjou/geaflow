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

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.apache.geaflow.api.context.RuntimeContext;
import org.apache.geaflow.api.graph.function.vc.IncVertexCentricComputeFunction.IncGraphComputeContext;
import org.apache.geaflow.api.graph.function.vc.base.IncGraphInferContext;
import org.apache.geaflow.api.graph.function.vc.base.IncVertexCentricFunction.GraphSnapShot;
import org.apache.geaflow.api.graph.function.vc.base.IncVertexCentricFunction.HistoricalGraph;
import org.apache.geaflow.api.graph.function.vc.base.IncVertexCentricFunction.MutableGraph;
import org.apache.geaflow.api.graph.function.vc.base.IncVertexCentricFunction.TemporaryGraph;
import org.apache.geaflow.api.graph.function.vc.base.VertexCentricFunction.EdgeQuery;
import org.apache.geaflow.api.graph.function.vc.base.VertexCentricFunction.VertexQuery;
import org.apache.geaflow.common.binary.BinaryString;
import org.apache.geaflow.common.config.Configuration;
import org.apache.geaflow.common.iterator.CloseableIterator;
import org.apache.geaflow.common.type.IType;
import org.apache.geaflow.common.type.Types;
import org.apache.geaflow.dsl.common.algo.AlgorithmRuntimeContext;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.data.RowEdge;
import org.apache.geaflow.dsl.common.data.RowVertex;
import org.apache.geaflow.dsl.common.data.impl.ObjectRow;
import org.apache.geaflow.dsl.common.types.GraphSchema;
import org.apache.geaflow.dsl.common.types.TableField;
import org.apache.geaflow.dsl.common.types.VertexType;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNConfig;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferPayload;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.vertex.IVertex;
import org.apache.geaflow.model.graph.vertex.impl.ValueVertex;
import org.apache.geaflow.state.pushdown.filter.IFilter;
import org.apache.geaflow.state.pushdown.filter.IVertexFilter;
import org.testng.annotations.Test;

public class GCNAlgorithmTest {

    @Test
    public void testGCNProcessCollectsInferenceResult() {
        TestAlgorithmRuntimeContext context = new TestAlgorithmRuntimeContext();
        context.addVertex(1L, new TestRowVertex(1L, ObjectRow.create(1, 2D)));
        context.addVertex(2L, new TestRowVertex(2L, ObjectRow.create(3, 4D)));
        context.addVertex(3L, new TestRowVertex(3L, ObjectRow.create(5, 6D)));
        context.addEdge(1L, new TestRowEdge(1L, 2L));
        context.addEdge(1L, new TestRowEdge(1L, 3L));
        context.inferResult = Map.of(
            "node_id", 1L,
            "embedding", List.of(0.1D, 0.2D),
            "prediction", 7L,
            "confidence", 0.8D
        );

        GCN gcn = new GCN();
        gcn.init(context, new Object[0]);
        gcn.process(context.vertices.get(1L), java.util.Optional.empty(), Collections.emptyIterator());

        ObjectRow row = (ObjectRow) context.takenRows.get(0);
        assertEquals(row.getFields()[0], 1L);
        assertEquals(row.getFields()[2], 7L);
        assertEquals(row.getFields()[3], 0.8D);
        assertNotNull(context.capturedPayload);
        assertEquals(context.capturedPayload.getCenter_node_id(), 1L);
        assertTrue(context.capturedPayload.getSampled_nodes().containsAll(List.of(1L, 2L, 3L)));
    }

    @Test
    public void testGCNProcessSkipsAfterFirstIteration() {
        TestAlgorithmRuntimeContext context = new TestAlgorithmRuntimeContext();
        context.iterationId = 2L;
        context.addVertex(1L, new TestRowVertex(1L, ObjectRow.create(1, 2D)));

        GCN gcn = new GCN();
        gcn.init(context, new Object[0]);
        gcn.process(context.vertices.get(1L), java.util.Optional.empty(), Collections.emptyIterator());

        assertTrue(context.takenRows.isEmpty());
        assertEquals(context.inferCallCount, 0);
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testGCNProcessRequiresDynamicContextMethods() {
        GCN gcn = new GCN();
        gcn.init(new MissingDynamicMethodsContext(buildGraphSchema()), new Object[0]);
        gcn.process(new TestRowVertex(1L, ObjectRow.create(1, 2D)),
            java.util.Optional.empty(), Collections.emptyIterator());
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testGCNInitRejectsInvalidParamCount() {
        new GCN().init(new TestAlgorithmRuntimeContext(), new Object[]{1});
    }

    @Test
    public void testGCNComputeFunctionCollectsVertexResult() {
        GCNCompute.GCNComputeFunction function = new GCNCompute.GCNComputeFunction(
            new GCNConfig(1, 2, GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS));
        TestIncGraphContext context = new TestIncGraphContext();
        context.addVertex(1L, new ValueVertex<>(1L, List.of(1D, 2D)));
        context.addVertex(2L, new ValueVertex<>(2L, List.of(3D, 4D)));
        context.addVertex(3L, new ValueVertex<>(3L, List.of(5D, 6D)));
        context.addEdges(1L, List.of(new TestEdge(1L, 2L), new TestEdge(1L, 3L)));
        context.inferResult = Map.of(
            "embedding", List.of(0.3D, 0.4D),
            "predicted_class", 9L,
            "confidence", 0.95D
        );

        function.init(context);
        function.evolve(1L, new TestTemporaryGraph(context.vertices.get(1L)));

        ValueVertex<Object, List<Object>> collected = (ValueVertex<Object, List<Object>>) context.collectedVertex;
        assertEquals(collected.getId(), 1L);
        assertEquals(((double[]) collected.getValue().get(0)), new double[]{0.3D, 0.4D});
        assertEquals(collected.getValue().get(1), 9L);
        assertEquals(collected.getValue().get(2), 0.95D);
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testGCNComputeFunctionRequiresInferContext() {
        GCNCompute.GCNComputeFunction function = new GCNCompute.GCNComputeFunction(
            new GCNConfig(1, 1, GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS));
        NonInferIncGraphContext context = new NonInferIncGraphContext();
        context.addVertex(1L, new ValueVertex<>(1L, List.of(1D, 2D)));

        function.init(context);
        function.evolve(1L, new TestTemporaryGraph(context.getVertex(1L)));
    }

    private static GraphSchema buildGraphSchema() {
        return new GraphSchema("g", List.of(
            new TableField("person", new VertexType(List.of(
                new TableField("~id", Types.LONG, false),
                new TableField("~label", Types.STRING, false),
                new TableField("f0", Types.INTEGER, true),
                new TableField("f1", Types.DOUBLE, true)
            )), false)
        ));
    }

    private static class TestAlgorithmRuntimeContext implements AlgorithmRuntimeContext<Object, Object> {

        private final Map<Object, TestRowVertex> vertices = new HashMap<>();
        private final Map<Object, List<RowEdge>> edges = new HashMap<>();
        private final List<Row> takenRows = new ArrayList<>();
        private final GraphSchema graphSchema = buildGraphSchema();
        private Object currentVertexId;
        private long iterationId = 1L;
        private int inferCallCount;
        private Object inferResult;
        private GCNInferPayload capturedPayload;

        void addVertex(Object id, TestRowVertex vertex) {
            vertices.put(id, vertex);
            currentVertexId = id;
        }

        void addEdge(Object nodeId, RowEdge edge) {
            edges.computeIfAbsent(nodeId, key -> new ArrayList<>()).add(edge);
        }

        public void setVertexId(Object vertexId) {
            this.currentVertexId = vertexId;
        }

        public RowVertex loadVertex() {
            return vertices.get(currentVertexId);
        }

        @Override
        public List<RowEdge> loadEdges(EdgeDirection direction) {
            return edges.getOrDefault(currentVertexId, Collections.emptyList());
        }

        @Override
        public void take(Row value) {
            takenRows.add(value);
        }

        public Object infer(Object... modelInputs) {
            inferCallCount++;
            capturedPayload = (GCNInferPayload) modelInputs[0];
            return inferResult;
        }

        @Override
        public long getCurrentIterationId() {
            return iterationId;
        }

        @Override
        public GraphSchema getGraphSchema() {
            return graphSchema;
        }

        @Override
        public Configuration getConfig() {
            return new Configuration();
        }

        @Override
        public CloseableIterator<RowEdge> loadEdgesIterator(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadEdgesIterator(IFilter filter) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<RowEdge> loadStaticEdges(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadStaticEdgesIterator(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadStaticEdgesIterator(IFilter filter) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<RowEdge> loadDynamicEdges(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadDynamicEdgesIterator(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadDynamicEdgesIterator(IFilter filter) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void sendMessage(Object vertexId, Object message) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void updateVertexValue(Row value) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void voteToTerminate(String terminationReason, Object voteValue) {
            throw new UnsupportedOperationException();
        }
    }

    private static final class MissingDynamicMethodsContext implements AlgorithmRuntimeContext<Object, Object> {

        private final GraphSchema graphSchema;

        private MissingDynamicMethodsContext(GraphSchema graphSchema) {
            this.graphSchema = graphSchema;
        }

        @Override
        public List<RowEdge> loadEdges(EdgeDirection direction) {
            return Collections.emptyList();
        }

        @Override
        public CloseableIterator<RowEdge> loadEdgesIterator(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadEdgesIterator(IFilter filter) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<RowEdge> loadStaticEdges(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadStaticEdgesIterator(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadStaticEdgesIterator(IFilter filter) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<RowEdge> loadDynamicEdges(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadDynamicEdgesIterator(EdgeDirection direction) {
            throw new UnsupportedOperationException();
        }

        @Override
        public CloseableIterator<RowEdge> loadDynamicEdgesIterator(IFilter filter) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void sendMessage(Object vertexId, Object message) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void updateVertexValue(Row value) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void take(Row value) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long getCurrentIterationId() {
            return 1L;
        }

        @Override
        public GraphSchema getGraphSchema() {
            return graphSchema;
        }

        @Override
        public Configuration getConfig() {
            return new Configuration();
        }

        @Override
        public void voteToTerminate(String terminationReason, Object voteValue) {
            throw new UnsupportedOperationException();
        }
    }

    private static class TestIncGraphContext implements IncGraphComputeContext<Object, List<Object>, Object, Object>,
        IncGraphInferContext<Object> {

        private final Map<Object, IVertex<Object, List<Object>>> vertices = new HashMap<>();
        private final Map<Object, List<IEdge<Object, Object>>> edges = new HashMap<>();
        private IVertex<Object, List<Object>> collectedVertex;
        private Object inferResult;

        void addVertex(Object id, IVertex<Object, List<Object>> vertex) {
            vertices.put(id, vertex);
        }

        void addEdges(Object id, List<IEdge<Object, Object>> vertexEdges) {
            edges.put(id, vertexEdges);
        }

        IVertex<Object, List<Object>> getVertex(Object id) {
            return vertices.get(id);
        }

        @Override
        public void collect(IVertex<Object, List<Object>> vertex) {
            this.collectedVertex = vertex;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Object infer(Object... modelInputs) {
            return inferResult;
        }

        @Override
        public HistoricalGraph<Object, List<Object>, Object> getHistoricalGraph() {
            return new HistoricalGraph<Object, List<Object>, Object>() {
                @Override
                public Long getLatestVersionId() {
                    return 1L;
                }

                @Override
                public List<Long> getAllVersionIds() {
                    return List.of(1L);
                }

                @Override
                public Map<Long, IVertex<Object, List<Object>>> getAllVertex() {
                    throw new UnsupportedOperationException();
                }

                @Override
                public Map<Long, IVertex<Object, List<Object>>> getAllVertex(List<Long> versions) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public Map<Long, IVertex<Object, List<Object>>> getAllVertex(List<Long> versions,
                                                                             IVertexFilter<Object, List<Object>> vertexFilter) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public GraphSnapShot<Object, List<Object>, Object> getSnapShot(long version) {
                    return new GraphSnapShot<Object, List<Object>, Object>() {
                        @Override
                        public long getVersion() {
                            return version;
                        }

                        @Override
                        public VertexQuery<Object, List<Object>> vertex() {
                            return new TestVertexQuery(vertices);
                        }

                        @Override
                        public EdgeQuery<Object, Object> edges() {
                            return new TestEdgeQuery(edges);
                        }
                    };
                }
            };
        }

        @Override
        public long getJobId() {
            return 1L;
        }

        @Override
        public long getIterationId() {
            return 1L;
        }

        @Override
        public RuntimeContext getRuntimeContext() {
            return null;
        }

        @Override
        public MutableGraph<Object, List<Object>, Object> getMutableGraph() {
            return null;
        }

        @Override
        public TemporaryGraph<Object, List<Object>, Object> getTemporaryGraph() {
            return null;
        }

        @Override
        public void sendMessage(Object vertexId, Object message) {
        }

        @Override
        public void sendMessageToNeighbors(Object message) {
        }

        @Override
        public void close() throws IOException {
        }
    }

    private static final class NonInferIncGraphContext implements IncGraphComputeContext<Object, List<Object>, Object, Object> {

        private final Map<Object, IVertex<Object, List<Object>>> vertices = new HashMap<>();

        void addVertex(Object id, IVertex<Object, List<Object>> vertex) {
            vertices.put(id, vertex);
        }

        IVertex<Object, List<Object>> getVertex(Object id) {
            return vertices.get(id);
        }

        @Override
        public void collect(IVertex<Object, List<Object>> vertex) {
        }

        @Override
        public long getJobId() {
            return 1L;
        }

        @Override
        public long getIterationId() {
            return 1L;
        }

        @Override
        public RuntimeContext getRuntimeContext() {
            return null;
        }

        @Override
        public MutableGraph<Object, List<Object>, Object> getMutableGraph() {
            return null;
        }

        @Override
        public TemporaryGraph<Object, List<Object>, Object> getTemporaryGraph() {
            return null;
        }

        @Override
        public HistoricalGraph<Object, List<Object>, Object> getHistoricalGraph() {
            return new TestIncGraphContext().getHistoricalGraph();
        }

        @Override
        public void sendMessage(Object vertexId, Object message) {
        }

        @Override
        public void sendMessageToNeighbors(Object message) {
        }
    }

    private static final class TestTemporaryGraph implements TemporaryGraph<Object, List<Object>, Object> {

        private final IVertex<Object, List<Object>> vertex;

        private TestTemporaryGraph(IVertex<Object, List<Object>> vertex) {
            this.vertex = vertex;
        }

        @Override
        public IVertex<Object, List<Object>> getVertex() {
            return vertex;
        }

        @Override
        public List<IEdge<Object, Object>> getEdges() {
            return Collections.emptyList();
        }

        @Override
        public void updateVertexValue(List<Object> value) {
            vertex.withValue(value);
        }
    }

    private static final class TestVertexQuery implements VertexQuery<Object, List<Object>> {

        private final Map<Object, IVertex<Object, List<Object>>> vertices;
        private Object currentId;

        private TestVertexQuery(Map<Object, IVertex<Object, List<Object>>> vertices) {
            this.vertices = vertices;
        }

        @Override
        public VertexQuery<Object, List<Object>> withId(Object vertexId) {
            this.currentId = vertexId;
            return this;
        }

        @Override
        public IVertex<Object, List<Object>> get() {
            return vertices.get(currentId);
        }

        @Override
        public IVertex<Object, List<Object>> get(IFilter vertexFilter) {
            return get();
        }
    }

    private static final class TestEdgeQuery implements EdgeQuery<Object, Object> {

        private final Map<Object, List<IEdge<Object, Object>>> edges;
        private Object currentId;

        private TestEdgeQuery(Map<Object, List<IEdge<Object, Object>>> edges) {
            this.edges = edges;
        }

        public TestEdgeQuery withId(Object vertexId) {
            this.currentId = vertexId;
            return this;
        }

        @Override
        public List<IEdge<Object, Object>> getEdges() {
            return edges.getOrDefault(currentId, Collections.emptyList());
        }

        @Override
        public List<IEdge<Object, Object>> getOutEdges() {
            return getEdges();
        }

        @Override
        public List<IEdge<Object, Object>> getInEdges() {
            return getEdges();
        }

        @Override
        public CloseableIterator<IEdge<Object, Object>> getEdges(IFilter<?> edgeFilter) {
            throw new UnsupportedOperationException();
        }
    }

    private static class TestRowVertex implements RowVertex {

        private Object id;
        private Row value;
        private String label = "person";
        private BinaryString binaryLabel;

        private TestRowVertex(Object id, Row value) {
            this.id = id;
            this.value = value;
        }

        @Override
        public void setValue(Row value) {
            this.value = value;
        }

        @Override
        public Object getId() {
            return id;
        }

        @Override
        public void setId(Object id) {
            this.id = id;
        }

        @Override
        public Row getValue() {
            return value;
        }

        @Override
        public IVertex<Object, Row> withValue(Row value) {
            this.value = value;
            return this;
        }

        @Override
        public IVertex<Object, Row> withLabel(String label) {
            this.label = label;
            return this;
        }

        @Override
        public IVertex<Object, Row> withTime(long time) {
            return this;
        }

        @Override
        public Object getField(int i, IType<?> type) {
            if (i == 0) {
                return id;
            }
            if (i == 1) {
                return label;
            }
            return value.getField(i - 2, type);
        }

        @Override
        public String getLabel() {
            return label;
        }

        @Override
        public void setLabel(String label) {
            this.label = label;
        }

        @Override
        public BinaryString getBinaryLabel() {
            return binaryLabel;
        }

        @Override
        public void setBinaryLabel(BinaryString label) {
            this.binaryLabel = label;
        }

        @Override
        public int compareTo(Object o) {
            return String.valueOf(id).compareTo(String.valueOf(((IVertex<?, ?>) o).getId()));
        }
    }

    private static final class TestRowEdge implements RowEdge {

        private Object srcId;
        private Object targetId;
        private Row value = Row.EMPTY;
        private String label = "knows";
        private BinaryString binaryLabel;
        private EdgeDirection direction = EdgeDirection.OUT;

        private TestRowEdge(Object srcId, Object targetId) {
            this.srcId = srcId;
            this.targetId = targetId;
        }

        @Override
        public void setValue(Row value) {
            this.value = value;
        }

        @Override
        public RowEdge withDirection(EdgeDirection direction) {
            this.direction = direction;
            return this;
        }

        @Override
        public RowEdge identityReverse() {
            return new TestRowEdge(targetId, srcId).withDirection(direction.reverse());
        }

        @Override
        public Object getSrcId() {
            return srcId;
        }

        @Override
        public void setSrcId(Object srcId) {
            this.srcId = srcId;
        }

        @Override
        public Object getTargetId() {
            return targetId;
        }

        @Override
        public void setTargetId(Object targetId) {
            this.targetId = targetId;
        }

        @Override
        public EdgeDirection getDirect() {
            return direction;
        }

        @Override
        public void setDirect(EdgeDirection direction) {
            this.direction = direction;
        }

        @Override
        public Row getValue() {
            return value;
        }

        @Override
        public IEdge<Object, Row> withValue(Row value) {
            this.value = value;
            return this;
        }

        @Override
        public IEdge<Object, Row> reverse() {
            return new TestRowEdge(targetId, srcId);
        }

        @Override
        public Object getField(int i, IType<?> type) {
            return value.getField(i, type);
        }

        @Override
        public String getLabel() {
            return label;
        }

        @Override
        public void setLabel(String label) {
            this.label = label;
        }

        @Override
        public BinaryString getBinaryLabel() {
            return binaryLabel;
        }

        @Override
        public void setBinaryLabel(BinaryString label) {
            this.binaryLabel = label;
        }
    }

    private static final class TestEdge implements IEdge<Object, Object> {

        private Object srcId;
        private Object targetId;

        private TestEdge(Object srcId, Object targetId) {
            this.srcId = srcId;
            this.targetId = targetId;
        }

        @Override
        public Object getSrcId() {
            return srcId;
        }

        @Override
        public void setSrcId(Object srcId) {
            this.srcId = srcId;
        }

        @Override
        public Object getTargetId() {
            return targetId;
        }

        @Override
        public void setTargetId(Object targetId) {
            this.targetId = targetId;
        }

        @Override
        public EdgeDirection getDirect() {
            return EdgeDirection.OUT;
        }

        @Override
        public void setDirect(EdgeDirection direction) {
        }

        @Override
        public Object getValue() {
            return null;
        }

        @Override
        public IEdge<Object, Object> withValue(Object value) {
            return this;
        }

        @Override
        public IEdge<Object, Object> reverse() {
            return new TestEdge(targetId, srcId);
        }
    }
}
