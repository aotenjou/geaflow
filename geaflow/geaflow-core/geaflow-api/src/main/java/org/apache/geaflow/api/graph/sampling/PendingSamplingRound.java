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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.geaflow.model.graph.edge.IEdge;

/** Bounded requester-side state retained between request and commit supersteps. */
public final class PendingSamplingRound<K, EV> implements Serializable {

    private final SamplingClock requestClock;
    private final K requesterId;
    private final Map<K, List<IEdge<K, EV>>> edgesByNeighbor;

    public PendingSamplingRound(SamplingClock requestClock, K requesterId,
                                Iterable<? extends IEdge<K, EV>> sampledEdges) {
        this.requestClock = NeighborStateRequest.requirePhase(requestClock, SamplingPhase.REQUEST);
        this.requesterId = Objects.requireNonNull(requesterId, "requesterId");
        Objects.requireNonNull(sampledEdges, "sampledEdges");
        this.edgesByNeighbor = new LinkedHashMap<>();
        for (IEdge<K, EV> edge : sampledEdges) {
            Objects.requireNonNull(edge, "edge");
            K neighborId = neighborId(requesterId, edge);
            edgesByNeighbor.computeIfAbsent(neighborId, ignored -> new ArrayList<>()).add(edge);
        }
    }

    private K neighborId(K vertexId, IEdge<K, EV> edge) {
        if (Objects.equals(vertexId, edge.getSrcId())) {
            return Objects.requireNonNull(edge.getTargetId(), "neighborId");
        }
        if (Objects.equals(vertexId, edge.getTargetId())) {
            return Objects.requireNonNull(edge.getSrcId(), "neighborId");
        }
        throw new IllegalArgumentException("sampled edge is not incident to requesterId=" + vertexId);
    }

    public SamplingClock getRequestClock() {
        return requestClock;
    }

    public K getRequesterId() {
        return requesterId;
    }

    public boolean isEmpty() {
        return edgesByNeighbor.isEmpty();
    }

    public List<K> getNeighborIds() {
        return Collections.unmodifiableList(new ArrayList<>(edgesByNeighbor.keySet()));
    }

    public Map<K, List<IEdge<K, EV>>> getEdgesByNeighbor() {
        Map<K, List<IEdge<K, EV>>> result = new LinkedHashMap<>();
        for (Map.Entry<K, List<IEdge<K, EV>>> entry : edgesByNeighbor.entrySet()) {
            result.put(entry.getKey(), Collections.unmodifiableList(entry.getValue()));
        }
        return Collections.unmodifiableMap(result);
    }

    public Map<K, NeighborStateRequest<K>> createRequests() {
        Map<K, NeighborStateRequest<K>> requests = new LinkedHashMap<>();
        for (K neighborId : edgesByNeighbor.keySet()) {
            requests.put(neighborId, new NeighborStateRequest<>(requestClock, requesterId));
        }
        return Collections.unmodifiableMap(requests);
    }

    public EmptySamplingRequest<K> createEmptyRequest() {
        if (!isEmpty()) {
            throw new IllegalStateException("only an empty sampling round uses an empty request");
        }
        return new EmptySamplingRequest<>(requestClock, requesterId);
    }
}
