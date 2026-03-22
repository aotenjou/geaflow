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

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.apache.geaflow.common.binary.BinaryString;
import org.apache.geaflow.common.type.IType;
import org.apache.geaflow.common.type.Types;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.data.RowVertex;
import org.apache.geaflow.dsl.common.data.impl.ObjectRow;
import org.apache.geaflow.dsl.common.types.GraphSchema;
import org.apache.geaflow.dsl.common.types.TableField;
import org.apache.geaflow.dsl.common.types.VertexType;
import org.testng.annotations.Test;

public class GCNComponentsTest {

    @Test
    public void testResultParserMap() {
        GCNResultParser parser = new GCNResultParser();
        GCNInferResult result = parser.parse(1L, Map.of(
            "node_id", 1L,
            "embedding", List.of(0.1D, 0.2D),
            "prediction", 2,
            "confidence", 0.9D
        ));
        assertEquals(result.getNodeId(), 1L);
        assertEquals(result.getPredictedClass(), 2L);
        assertEquals(result.getConfidence(), 0.9D);
        assertEquals(result.getEmbedding(), new double[]{0.1D, 0.2D});
    }

    @Test
    public void testResultParserListArrayNullAndPythonException() {
        GCNResultParser parser = new GCNResultParser();

        GCNInferResult listResult = parser.parse(1L, List.of(List.of(1.0D, 2.0D), 3, 0.5D));
        assertEquals(listResult.getNodeId(), 1L);
        assertEquals(listResult.getEmbedding(), new double[]{1.0D, 2.0D});
        assertEquals(listResult.getPredictedClass(), 3L);
        assertEquals(listResult.getConfidence(), 0.5D);

        Object[] arrayResult = new Object[]{2L, new Double[]{3.0D, null, 4.0D}, 5L, 0.7D};
        GCNInferResult parsedArray = parser.parse(1L, arrayResult);
        assertEquals(parsedArray.getNodeId(), 2L);
        assertEquals(parsedArray.getEmbedding(), new double[]{3.0D, 0D, 4.0D});
        assertEquals(parsedArray.getPredictedClass(), 5L);
        assertEquals(parsedArray.getConfidence(), 0.7D);

        GCNInferResult nullResult = parser.parse(7L, null);
        assertEquals(nullResult.getNodeId(), 7L);
        assertEquals(nullResult.getEmbedding(), new double[0]);
        assertEquals(nullResult.getPredictedClass(), -1L);
        assertEquals(nullResult.getConfidence(), 0D);
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testResultParserThrowsOnPythonExceptionMarker() {
        new GCNResultParser().parse(1L, "python_exception:test");
    }

    @Test
    public void testFeatureCollectorSupportsMultipleInputShapes() {
        GCNFeatureCollector collector = new GCNFeatureCollector();

        assertEquals(collector.collectFromValue(List.of(1, 2, 3), 5),
            new double[]{1D, 2D, 3D, 0D, 0D});
        assertEquals(collector.collectFromValue(new Object[]{1, "2.5"}, 4),
            new double[]{1D, 2.5D, 0D, 0D});
        assertEquals(collector.collectFromValue(ObjectRow.create(4, 5.5D), 4),
            new double[]{4D, 5.5D, 0D, 0D});
        assertEquals(collector.collectFromValue(6, 3),
            new double[]{6D, 0D, 0D});
        assertEquals(collector.collectFromValue(null, 2),
            new double[]{0D, 0D});
    }

    @Test
    public void testFeatureCollectorFromRowVertexHonorsSchemaAndPads() {
        GraphSchema graphSchema = new GraphSchema("g", List.of(
            new TableField("person", new VertexType(List.of(
                new TableField("~id", Types.LONG, false),
                new TableField("~label", Types.STRING, false),
                new TableField("f0", Types.INTEGER, true),
                new TableField("f1", Types.DOUBLE, true)
            )), false)
        ));
        TestRowVertex vertex = new TestRowVertex(1L, ObjectRow.create(10, 20.5D));

        assertEquals(new GCNFeatureCollector().collectFromRowVertex(vertex, graphSchema, 4),
            new double[]{10D, 20.5D, 0D, 0D});
    }

    @Test
    public void testSubgraphBuilderAddsSelfLoopAndSamplesDeterministically() {
        GCNConfig config = new GCNConfig(2, 2, GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS);
        GCNSubgraphBuilder builder = new GCNSubgraphBuilder(config);
        GCNInferPayload payload = builder.build(1L, new GCNSubgraphBuilder.GraphAdapter() {
            @Override
            public List<Object> loadNeighbors(Object nodeId) {
                if (nodeId.equals(1L)) {
                    return Arrays.asList(2L, 3L);
                }
                if (nodeId.equals(2L)) {
                    return Collections.singletonList(4L);
                }
                return Collections.emptyList();
            }

            @Override
            public double[] loadFeatures(Object nodeId) {
                return new double[]{((Number) nodeId).doubleValue()};
            }
        });
        assertEquals(payload.getCenter_node_id(), 1L);
        assertEquals(payload.getSampled_nodes().size(), 4);
        assertTrue(payload.getSampled_nodes().containsAll(Arrays.asList(1L, 2L, 3L, 4L)));
        assertEquals(payload.getNode_features().size(), 4);
        assertEquals(payload.getEdge_index()[0].length, 7);
        assertNotNull(payload.getEdge_index());
    }

    @Test
    public void testSubgraphBuilderRespectsSampleLimitAndCanDisableSelfLoop() {
        GCNConfig config = new GCNConfig(1, 1, GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS,
            false, 4, 123L);
        GCNSubgraphBuilder builder = new GCNSubgraphBuilder(config);
        GCNInferPayload payload = builder.build(1L, new GCNSubgraphBuilder.GraphAdapter() {
            @Override
            public List<Object> loadNeighbors(Object nodeId) {
                return Arrays.asList(2L, 3L, 4L);
            }

            @Override
            public double[] loadFeatures(Object nodeId) {
                return new double[]{((Number) nodeId).doubleValue()};
            }
        });

        assertEquals(payload.getSampled_nodes().size(), 2);
        assertEquals(payload.getEdge_index()[0].length, 1);
        assertTrue(payload.getSampled_nodes().contains(1L));
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testConfigRejectsBlankTransformClass() {
        new GCNConfig(1, 1, "   ");
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testConfigRejectsNonPositiveHopCount() {
        new GCNConfig(0, 1, GCNConfig.DEFAULT_PYTHON_TRANSFORM_CLASS);
    }

    private static class TestRowVertex implements RowVertex {

        private final Object id;
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
        public Row getValue() {
            return value;
        }

        @Override
        public void setId(Object id) {
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
        public org.apache.geaflow.model.graph.vertex.IVertex<Object, Row> withValue(Row value) {
            this.value = value;
            return this;
        }

        @Override
        public org.apache.geaflow.model.graph.vertex.IVertex<Object, Row> withLabel(String label) {
            this.label = label;
            return this;
        }

        @Override
        public org.apache.geaflow.model.graph.vertex.IVertex<Object, Row> withTime(long time) {
            return this;
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
            return String.valueOf(id).compareTo(String.valueOf(((RowVertex) o).getId()));
        }
    }
}
