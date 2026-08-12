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

package org.apache.geaflow.api.graph.sampling;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.edge.impl.ValueEdge;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PendingSamplingRoundTest {

    @Test
    public void testGroupsParallelEdgesAndOrdersResponsesBySampledNeighbor() {
        PendingSamplingRound<Long, String> pending = new PendingSamplingRound<>(requestClock(), 1L,
            Arrays.asList(edge(1L, 2L, "first"), edge(1L, 2L, "parallel"),
                edge(1L, 3L, "third")));

        Assert.assertEquals(pending.getNeighborIds(), Arrays.asList(2L, 3L));
        Assert.assertEquals(pending.getEdgesByNeighbor().get(2L).size(), 2);
        Assert.assertEquals(pending.createRequests().keySet(),
            new java.util.LinkedHashSet<>(Arrays.asList(2L, 3L)));

        SamplingResponseCollector<Long, String> collector = new SamplingResponseCollector<>(pending);
        collector.add(new NeighborStateResponse<>(requestClock().responseClock(), 1L, 3L, "three"));
        collector.add(new NeighborStateResponse<>(requestClock().responseClock(), 1L, 2L, "two"));

        List<NeighborStateResponse<Long, String>> responses = collector.getResponses();
        Assert.assertEquals(responses.get(0).getResponderId(), Long.valueOf(2L));
        Assert.assertEquals(responses.get(1).getResponderId(), Long.valueOf(3L));
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testRejectsMissingResponseAtCommit() {
        PendingSamplingRound<Long, String> pending = new PendingSamplingRound<>(requestClock(), 1L,
            Arrays.asList(edge(1L, 2L, "first"), edge(1L, 3L, "second")));
        SamplingResponseCollector<Long, String> collector = new SamplingResponseCollector<>(pending);
        collector.add(new NeighborStateResponse<>(requestClock().responseClock(), 1L, 2L, "two"));
        collector.validateComplete();
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testRejectsDuplicateResponse() {
        PendingSamplingRound<Long, String> pending = new PendingSamplingRound<>(requestClock(), 1L,
            Collections.singletonList(edge(1L, 2L, "first")));
        SamplingResponseCollector<Long, String> collector = new SamplingResponseCollector<>(pending);
        NeighborStateResponse<Long, String> response = new NeighborStateResponse<>(
            requestClock().responseClock(), 1L, 2L, "two");
        collector.add(response);
        collector.add(response);
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testRejectsResponseFromAnotherSession() {
        PendingSamplingRound<Long, String> pending = new PendingSamplingRound<>(requestClock(), 1L,
            Collections.singletonList(edge(1L, 2L, "first")));
        SamplingResponseCollector<Long, String> collector = new SamplingResponseCollector<>(pending);
        SamplingClock stale = new SamplingClock(7L, 12L, 1, SamplingPhase.RESPOND);
        collector.add(new NeighborStateResponse<>(stale, 1L, 2L, "two"));
    }

    @Test
    public void testEmptyRoundUsesTwoPhaseSelfBarrier() {
        PendingSamplingRound<Long, String> pending = new PendingSamplingRound<>(requestClock(), 1L,
            Collections.emptyList());
        SamplingResponseCollector<Long, String> collector = new SamplingResponseCollector<>(pending);

        EmptySamplingRequest<Long> request = pending.createEmptyRequest();
        Assert.assertEquals(request.getVertexId(), Long.valueOf(1L));
        collector.addEmpty(new EmptySamplingResponse<>(request.getClock().responseClock(), 1L));
        Assert.assertTrue(collector.isComplete());
        Assert.assertTrue(collector.getResponses().isEmpty());
    }

    @Test
    public void testProtocolStateIsSerializable() throws Exception {
        PendingSamplingRound<Long, String> pending = new PendingSamplingRound<>(requestClock(), 1L,
            Collections.singletonList(edge(1L, 2L, "first")));
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
            output.writeObject(pending);
            output.writeObject(pending.createRequests().get(2L));
        }
        Assert.assertTrue(bytes.size() > 0);
    }

    private SamplingClock requestClock() {
        return new SamplingClock(7L, 11L, 1, SamplingPhase.REQUEST);
    }

    private IEdge<Long, String> edge(long source, long target, String value) {
        return new ValueEdge<>(source, target, value, EdgeDirection.OUT);
    }
}
