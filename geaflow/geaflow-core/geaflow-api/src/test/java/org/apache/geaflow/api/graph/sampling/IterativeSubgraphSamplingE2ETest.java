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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.edge.impl.ValueEdge;
import org.apache.geaflow.state.sampling.DeterministicNeighborSampler;
import org.testng.Assert;
import org.testng.annotations.Test;

public class IterativeSubgraphSamplingE2ETest {

    private static final long SNAPSHOT_VERSION = 7L;
    private static final long SESSION_ID = 11L;
    private static final long START_ITERATION_ID = 21L;

    @Test
    public void testRoutesTwoHopSamplingAcrossAllBspPhases() {
        Map<Long, List<IEdge<Long, Integer>>> adjacency = new LinkedHashMap<>();
        adjacency.put(1L, Arrays.asList(edge(1L, 2L), edge(1L, 3L)));
        adjacency.put(2L, Collections.singletonList(edge(2L, 4L)));
        adjacency.put(3L, Collections.singletonList(edge(3L, 5L)));
        adjacency.put(4L, Collections.emptyList());
        adjacency.put(5L, Collections.emptyList());
        SubgraphSamplingSpec spec = new SubgraphSamplingSpec(
            2, -1, EdgeDirection.OUT, 100L, 17L);
        InMemorySamplingScheduler scheduler = new InMemorySamplingScheduler(adjacency, spec);

        List<SamplingPhase> phases = scheduler.run();

        Assert.assertEquals(phases, Arrays.asList(
            SamplingPhase.REQUEST,
            SamplingPhase.RESPOND,
            SamplingPhase.COMMIT_AND_REQUEST,
            SamplingPhase.RESPOND,
            SamplingPhase.COMPLETE));
        Assert.assertEquals(scheduler.getRootPayloads(), Arrays.asList(
            vertexIds(1L, 2L, 3L),
            vertexIds(1L, 2L, 3L, 4L, 5L)));
        Assert.assertEquals(scheduler.getNeighborRequestCount(), 8);
        Assert.assertEquals(scheduler.getNeighborResponseCount(), 8);
        Assert.assertEquals(scheduler.getEmptyRequestCount(), 4);
        Assert.assertEquals(scheduler.getEmptyResponseCount(), 4);
        Assert.assertEquals(scheduler.getCommitCount(), 10);
        Assert.assertEquals(scheduler.getSamplingCallCount(), 10);
        scheduler.assertComplete();
    }

    @Test
    public void testDrivesCompleteThreeHopMessageSchedule() {
        Map<Long, List<IEdge<Long, Integer>>> adjacency = new LinkedHashMap<>();
        adjacency.put(1L, Arrays.asList(edge(1L, 2L), edge(1L, 3L)));
        adjacency.put(2L, Collections.singletonList(edge(2L, 4L)));
        adjacency.put(3L, Collections.singletonList(edge(3L, 5L)));
        adjacency.put(4L, Collections.singletonList(edge(4L, 6L)));
        adjacency.put(5L, Collections.singletonList(edge(5L, 7L)));
        adjacency.put(6L, Collections.emptyList());
        adjacency.put(7L, Collections.emptyList());
        SubgraphSamplingSpec spec = new SubgraphSamplingSpec(
            3, -1, EdgeDirection.OUT, 100L, 17L);
        InMemorySamplingScheduler scheduler = new InMemorySamplingScheduler(adjacency, spec);

        Assert.assertEquals(scheduler.run(), Arrays.asList(
            SamplingPhase.REQUEST,
            SamplingPhase.RESPOND,
            SamplingPhase.COMMIT_AND_REQUEST,
            SamplingPhase.RESPOND,
            SamplingPhase.COMMIT_AND_REQUEST,
            SamplingPhase.RESPOND,
            SamplingPhase.COMPLETE));
        Assert.assertEquals(scheduler.getRootPayloads(), Arrays.asList(
            vertexIds(1L, 2L, 3L),
            vertexIds(1L, 2L, 3L, 4L, 5L),
            vertexIds(1L, 2L, 3L, 4L, 5L, 6L, 7L)));
        Assert.assertEquals(scheduler.getRootMessageTrace(), Arrays.asList(
            "request[1] 1->2",
            "request[1] 1->3",
            "response[1] 2->1",
            "response[1] 3->1",
            "request[2] 1->2",
            "request[2] 1->3",
            "response[2] 2->1",
            "response[2] 3->1",
            "request[3] 1->2",
            "request[3] 1->3",
            "response[3] 2->1",
            "response[3] 3->1"));
        Assert.assertEquals(new LinkedHashSet<>(scheduler.getRootSamplingVersions()).size(), 3,
            "each hop must use a distinct sampling version");

        // Every vertex participates in every hop, including explicit empty rounds for leaves.
        Assert.assertEquals(scheduler.getNeighborRequestCount(), 18);
        Assert.assertEquals(scheduler.getNeighborResponseCount(), 18);
        Assert.assertEquals(scheduler.getEmptyRequestCount(), 6);
        Assert.assertEquals(scheduler.getEmptyResponseCount(), 6);
        Assert.assertEquals(scheduler.getCommitCount(), 21);
        Assert.assertEquals(scheduler.getSamplingCallCount(), 21);
        scheduler.assertComplete();
    }

    private static Set<Long> vertexIds(Long... ids) {
        return new LinkedHashSet<>(Arrays.asList(ids));
    }

    private static IEdge<Long, Integer> edge(long source, long target) {
        return new ValueEdge<>(source, target, 1, EdgeDirection.OUT);
    }

    private static final class InMemorySamplingScheduler {

        private final Map<Long, List<IEdge<Long, Integer>>> adjacency;
        private final SubgraphSamplingSpec spec;
        private final Map<Long, IterativeSamplingState<Long, Integer, Set<Long>>> states =
            new LinkedHashMap<>();
        private final List<Set<Long>> rootPayloads = new ArrayList<>();
        private final List<String> rootMessageTrace = new ArrayList<>();
        private final List<Long> rootSamplingVersions = new ArrayList<>();
        private int neighborRequestCount;
        private int neighborResponseCount;
        private int emptyRequestCount;
        private int emptyResponseCount;
        private int commitCount;
        private int samplingCallCount;

        private InMemorySamplingScheduler(Map<Long, List<IEdge<Long, Integer>>> adjacency,
                                          SubgraphSamplingSpec spec) {
            this.adjacency = adjacency;
            this.spec = spec;
            for (Long vertexId : adjacency.keySet()) {
                states.put(vertexId, new IterativeSamplingState<>(
                    SNAPSHOT_VERSION, SESSION_ID, vertexIds(vertexId)));
            }
        }

        private List<SamplingPhase> run() {
            List<SamplingPhase> phases = new ArrayList<>();
            Map<Long, List<SamplingMessage>> inbox = Collections.emptyMap();
            long iterations = SamplingClock.requiredIterations(spec.getHops());
            for (long offset = 0L; offset < iterations; offset++) {
                long iterationId = START_ITERATION_ID + offset;
                SamplingClock clock = SamplingClock.forIteration(SNAPSHOT_VERSION, SESSION_ID,
                    spec.getHops(), START_ITERATION_ID, iterationId);
                phases.add(clock.getPhase());
                switch (clock.getPhase()) {
                    case REQUEST:
                        Assert.assertTrue(inbox.isEmpty());
                        inbox = startRounds(clock);
                        break;
                    case RESPOND:
                        inbox = respond(clock, inbox);
                        break;
                    case COMMIT_AND_REQUEST:
                        commit(clock, inbox);
                        inbox = startRounds(clock.nextRequestClock());
                        break;
                    case COMPLETE:
                        commit(clock, inbox);
                        inbox = Collections.emptyMap();
                        break;
                    default:
                        throw new IllegalStateException("unsupported sampling phase: "
                            + clock.getPhase());
                }
            }
            Assert.assertTrue(inbox.isEmpty());
            return phases;
        }

        private Map<Long, List<SamplingMessage>> startRounds(SamplingClock requestClock) {
            Map<Long, List<SamplingMessage>> requests = new LinkedHashMap<>();
            for (Map.Entry<Long, IterativeSamplingState<Long, Integer, Set<Long>>> entry
                : states.entrySet()) {
                Long requesterId = entry.getKey();
                List<IEdge<Long, Integer>> sampled = DeterministicNeighborSampler.sample(
                    requesterId, adjacency.get(requesterId), spec.getDirection(), spec.getFanout(),
                    Comparator.naturalOrder(), spec.getMaxReturnedEdges(), spec.getSeed(),
                    requestClock.getSamplingVersion());
                samplingCallCount++;
                PendingSamplingRound<Long, Integer> pending = new PendingSamplingRound<>(
                    requestClock, requesterId, sampled);
                entry.getValue().startRound(pending);
                if (requesterId.equals(1L)) {
                    rootSamplingVersions.add(requestClock.getSamplingVersion());
                }
                if (pending.isEmpty()) {
                    route(requests, requesterId, pending.createEmptyRequest());
                    emptyRequestCount++;
                } else {
                    for (Map.Entry<Long, NeighborStateRequest<Long>> request
                        : pending.createRequests().entrySet()) {
                        route(requests, request.getKey(), request.getValue());
                        if (requesterId.equals(1L)) {
                            rootMessageTrace.add("request[" + requestClock.getHop() + "] "
                                + requesterId + "->" + request.getKey());
                        }
                        neighborRequestCount++;
                    }
                }
            }
            return requests;
        }

        private Map<Long, List<SamplingMessage>> respond(
            SamplingClock responseClock, Map<Long, List<SamplingMessage>> requests) {
            Map<Long, List<SamplingMessage>> responses = new LinkedHashMap<>();
            for (Map.Entry<Long, List<SamplingMessage>> inbox : requests.entrySet()) {
                Long responderId = inbox.getKey();
                IterativeSamplingState<Long, Integer, Set<Long>> responder = states.get(responderId);
                Assert.assertNotNull(responder, "message routed to an unknown vertex");
                for (SamplingMessage message : inbox.getValue()) {
                    Assert.assertTrue(message.getClock().isSameRound(responseClock));
                    if (message instanceof NeighborStateRequest) {
                        NeighborStateRequest<Long> request = (NeighborStateRequest<Long>) message;
                        route(responses, request.getRequesterId(),
                            responder.respond(responderId, request));
                        if (request.getRequesterId().equals(1L)) {
                            rootMessageTrace.add("response[" + responseClock.getHop() + "] "
                                + responderId + "->" + request.getRequesterId());
                        }
                        neighborResponseCount++;
                    } else if (message instanceof EmptySamplingRequest) {
                        EmptySamplingRequest<Long> request = (EmptySamplingRequest<Long>) message;
                        Assert.assertEquals(request.getVertexId(), responderId);
                        route(responses, responderId, new EmptySamplingResponse<>(
                            request.getClock().responseClock(), responderId));
                        emptyResponseCount++;
                    } else {
                        throw new IllegalStateException("unexpected sampling request: " + message);
                    }
                }
            }
            return responses;
        }

        private void commit(SamplingClock commitClock,
                            Map<Long, List<SamplingMessage>> responses) {
            Map<Long, Set<Long>> nextPayloads = new LinkedHashMap<>();
            for (Map.Entry<Long, IterativeSamplingState<Long, Integer, Set<Long>>> entry
                : states.entrySet()) {
                Long requesterId = entry.getKey();
                PendingSamplingRound<Long, Integer> pending = entry.getValue().getPendingRound();
                SamplingResponseCollector<Long, Set<Long>> collector =
                    new SamplingResponseCollector<>(pending);
                for (SamplingMessage message
                    : responses.getOrDefault(requesterId, Collections.emptyList())) {
                    if (message instanceof NeighborStateResponse) {
                        collector.add((NeighborStateResponse<Long, Set<Long>>) message);
                    } else if (message instanceof EmptySamplingResponse) {
                        collector.addEmpty((EmptySamplingResponse<Long>) message);
                    } else {
                        throw new IllegalStateException("unexpected sampling response: " + message);
                    }
                }
                Set<Long> nextPayload = vertexIds(requesterId);
                for (NeighborStateResponse<Long, Set<Long>> response : collector.getResponses()) {
                    nextPayload.addAll(response.getPayload());
                }
                nextPayloads.put(requesterId, nextPayload);
            }
            for (Map.Entry<Long, Set<Long>> payload : nextPayloads.entrySet()) {
                states.get(payload.getKey()).commit(commitClock, payload.getValue());
                commitCount++;
            }
            rootPayloads.add(new LinkedHashSet<>(states.get(1L).getCommittedPayload()));
        }

        private void route(Map<Long, List<SamplingMessage>> messages, Long destination,
                           SamplingMessage message) {
            messages.computeIfAbsent(destination, ignored -> new ArrayList<>()).add(message);
        }

        private void assertComplete() {
            for (IterativeSamplingState<Long, Integer, Set<Long>> state : states.values()) {
                Assert.assertEquals(state.getCompletedHop(), spec.getHops());
                Assert.assertNull(state.getPendingRound());
            }
        }

        private List<Set<Long>> getRootPayloads() {
            return rootPayloads;
        }

        private List<String> getRootMessageTrace() {
            return rootMessageTrace;
        }

        private List<Long> getRootSamplingVersions() {
            return rootSamplingVersions;
        }

        private int getNeighborRequestCount() {
            return neighborRequestCount;
        }

        private int getNeighborResponseCount() {
            return neighborResponseCount;
        }

        private int getEmptyRequestCount() {
            return emptyRequestCount;
        }

        private int getEmptyResponseCount() {
            return emptyResponseCount;
        }

        private int getCommitCount() {
            return commitCount;
        }

        private int getSamplingCallCount() {
            return samplingCallCount;
        }
    }
}
