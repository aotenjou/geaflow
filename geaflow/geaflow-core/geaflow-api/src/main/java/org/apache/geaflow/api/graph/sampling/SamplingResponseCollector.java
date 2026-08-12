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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Validates and orders all responses for one pending sampling round. */
public final class SamplingResponseCollector<K, P> {

    private final SamplingClock requestClock;
    private final K requesterId;
    private final List<K> expectedNeighbors;
    private final Map<K, NeighborStateResponse<K, P>> responses = new LinkedHashMap<>();
    private boolean emptyResponseReceived;

    public SamplingResponseCollector(PendingSamplingRound<K, ?> pending) {
        Objects.requireNonNull(pending, "pending");
        this.requestClock = pending.getRequestClock();
        this.requesterId = pending.getRequesterId();
        this.expectedNeighbors = pending.getNeighborIds();
    }

    public void add(NeighborStateResponse<K, P> response) {
        Objects.requireNonNull(response, "response");
        requireResponseRound(response.getClock());
        if (!Objects.equals(requesterId, response.getRequesterId())) {
            throw new IllegalArgumentException("sampling response requester does not match pending round");
        }
        K responderId = response.getResponderId();
        if (!expectedNeighbors.contains(responderId)) {
            throw new IllegalArgumentException("sampling response came from an unrequested neighbor: "
                + responderId);
        }
        if (responses.putIfAbsent(responderId, response) != null) {
            throw new IllegalStateException("duplicate sampling response from neighbor: " + responderId);
        }
    }

    public void addEmpty(EmptySamplingResponse<K> response) {
        Objects.requireNonNull(response, "response");
        requireResponseRound(response.getClock());
        if (!expectedNeighbors.isEmpty()) {
            throw new IllegalStateException("non-empty sampling round cannot accept an empty response");
        }
        if (!Objects.equals(requesterId, response.getVertexId())) {
            throw new IllegalArgumentException("empty sampling response vertex does not match requester");
        }
        if (emptyResponseReceived) {
            throw new IllegalStateException("duplicate empty sampling response");
        }
        emptyResponseReceived = true;
    }

    private void requireResponseRound(SamplingClock responseClock) {
        NeighborStateRequest.requirePhase(responseClock, SamplingPhase.RESPOND);
        if (!requestClock.isSameRound(responseClock)) {
            throw new IllegalArgumentException("sampling response clock does not match pending round");
        }
    }

    public boolean isComplete() {
        return expectedNeighbors.isEmpty() ? emptyResponseReceived
            : responses.size() == expectedNeighbors.size();
    }

    public void validateComplete() {
        if (!isComplete()) {
            List<K> missing = new ArrayList<>(expectedNeighbors);
            missing.removeAll(responses.keySet());
            throw new IllegalStateException("sampling responses are incomplete, requesterId="
                + requesterId + ", missing=" + missing);
        }
    }

    public List<NeighborStateResponse<K, P>> getResponses() {
        validateComplete();
        List<NeighborStateResponse<K, P>> ordered = new ArrayList<>(expectedNeighbors.size());
        for (K neighborId : expectedNeighbors) {
            ordered.add(responses.get(neighborId));
        }
        return Collections.unmodifiableList(ordered);
    }
}
