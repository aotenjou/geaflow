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

import java.util.Objects;

/** Request for a neighbor's state committed at the preceding hop. */
public final class NeighborStateRequest<K> implements SamplingMessage {

    private final SamplingClock clock;
    private final K requesterId;

    public NeighborStateRequest(SamplingClock clock, K requesterId) {
        this.clock = requirePhase(clock, SamplingPhase.REQUEST);
        this.requesterId = Objects.requireNonNull(requesterId, "requesterId");
    }

    static SamplingClock requirePhase(SamplingClock clock, SamplingPhase phase) {
        Objects.requireNonNull(clock, "clock");
        if (clock.getPhase() != phase) {
            throw new IllegalArgumentException("sampling message requires phase " + phase);
        }
        return clock;
    }

    @Override
    public SamplingClock getClock() {
        return clock;
    }

    public K getRequesterId() {
        return requesterId;
    }
}
