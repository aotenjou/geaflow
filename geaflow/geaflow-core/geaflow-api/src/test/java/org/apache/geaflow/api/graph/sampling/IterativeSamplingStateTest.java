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

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.edge.impl.ValueEdge;
import org.testng.Assert;
import org.testng.annotations.Test;

public class IterativeSamplingStateTest {

    @Test
    public void testAllVerticesCommitTwoHopsWithoutAccumulatingSubgraph() {
        Map<Long, List<IEdge<Long, Integer>>> adjacency = new LinkedHashMap<>();
        adjacency.put(1L, Arrays.asList(edge(1L, 2L)));
        adjacency.put(2L, Arrays.asList(edge(2L, 1L), edge(2L, 3L)));
        adjacency.put(3L, Arrays.asList(edge(3L, 2L)));
        Map<Long, IterativeSamplingState<Long, Integer, Integer>> states = new LinkedHashMap<>();
        states.put(1L, new IterativeSamplingState<>(7L, 11L, 1));
        states.put(2L, new IterativeSamplingState<>(7L, 11L, 2));
        states.put(3L, new IterativeSamplingState<>(7L, 11L, 3));

        SamplingClock request = SamplingClock.forIteration(7L, 11L, 2, 1L, 1L);
        for (int hop = 1; hop <= 2; hop++) {
            Map<Long, Integer> nextPayloads = new LinkedHashMap<>();
            for (Map.Entry<Long, IterativeSamplingState<Long, Integer, Integer>> entry
                : states.entrySet()) {
                Long requesterId = entry.getKey();
                PendingSamplingRound<Long, Integer> pending = new PendingSamplingRound<>(request,
                    requesterId, adjacency.get(requesterId));
                entry.getValue().startRound(pending);
                Assert.assertTrue(pending.getNeighborIds().size() <= 2);

                SamplingResponseCollector<Long, Integer> collector =
                    new SamplingResponseCollector<>(pending);
                for (Map.Entry<Long, NeighborStateRequest<Long>> outbound
                    : pending.createRequests().entrySet()) {
                    collector.add(states.get(outbound.getKey()).respond(outbound.getKey(),
                        outbound.getValue()));
                }
                int next = collector.getResponses().stream()
                    .mapToInt(NeighborStateResponse::getPayload).sum();
                nextPayloads.put(requesterId, next);
            }

            SamplingClock commit = SamplingClock.forIteration(7L, 11L, 2, 1L, hop * 2L + 1L);
            for (Map.Entry<Long, Integer> payload : nextPayloads.entrySet()) {
                states.get(payload.getKey()).commit(commit, payload.getValue());
            }
            if (hop < 2) {
                request = commit.nextRequestClock();
            }
        }

        Assert.assertEquals(states.get(1L).getCommittedPayload(), Integer.valueOf(4));
        Assert.assertEquals(states.get(2L).getCommittedPayload(), Integer.valueOf(4));
        Assert.assertEquals(states.get(3L).getCommittedPayload(), Integer.valueOf(4));
        for (IterativeSamplingState<Long, Integer, Integer> state : states.values()) {
            Assert.assertEquals(state.getCompletedHop(), 2);
            Assert.assertNull(state.getPendingRound());
        }
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testCannotServeNextHopBeforePreviousHopCommit() {
        IterativeSamplingState<Long, Integer, Integer> state =
            new IterativeSamplingState<>(7L, 11L, 1);
        NeighborStateRequest<Long> request = new NeighborStateRequest<>(
            new SamplingClock(7L, 11L, 2, SamplingPhase.REQUEST), 2L);
        state.respond(1L, request);
    }

    private IEdge<Long, Integer> edge(long source, long target) {
        return new ValueEdge<>(source, target, 1, EdgeDirection.OUT);
    }
}
