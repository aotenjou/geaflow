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
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.assertNotNull;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
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
}
