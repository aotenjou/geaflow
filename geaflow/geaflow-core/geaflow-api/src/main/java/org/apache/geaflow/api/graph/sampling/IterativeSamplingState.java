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
import java.util.Objects;

/** Per-vertex committed payload and bounded in-flight state for one sampling session. */
public final class IterativeSamplingState<K, EV, P> implements Serializable {

    private final long snapshotVersion;
    private final long sessionId;
    private int completedHop;
    private P committedPayload;
    private PendingSamplingRound<K, EV> pendingRound;

    public IterativeSamplingState(long snapshotVersion, long sessionId, P initialPayload) {
        this.snapshotVersion = snapshotVersion;
        this.sessionId = sessionId;
        this.committedPayload = initialPayload;
    }

    public void startRound(PendingSamplingRound<K, EV> pending) {
        Objects.requireNonNull(pending, "pending");
        SamplingClock clock = pending.getRequestClock();
        requireSession(clock);
        if (pendingRound != null) {
            throw new IllegalStateException("a sampling round is already pending");
        }
        if (clock.getHop() != completedHop + 1) {
            throw new IllegalArgumentException("sampling request hop does not follow committed state");
        }
        this.pendingRound = pending;
    }

    public NeighborStateResponse<K, P> respond(K responderId, NeighborStateRequest<K> request) {
        Objects.requireNonNull(request, "request");
        requireSession(request.getClock());
        if (request.getClock().getHop() != completedHop + 1) {
            throw new IllegalStateException("requested payload is not committed for the preceding hop");
        }
        return new NeighborStateResponse<>(request.getClock().responseClock(),
            request.getRequesterId(), responderId, committedPayload);
    }

    public void commit(SamplingClock commitClock, P payload) {
        Objects.requireNonNull(commitClock, "commitClock");
        requireSession(commitClock);
        if (commitClock.getPhase() != SamplingPhase.COMMIT_AND_REQUEST
            && commitClock.getPhase() != SamplingPhase.COMPLETE) {
            throw new IllegalArgumentException("sampling state can only commit in a commit phase");
        }
        if (pendingRound == null || !pendingRound.getRequestClock().isSameRound(commitClock)) {
            throw new IllegalStateException("sampling commit does not match the pending round");
        }
        this.completedHop = commitClock.getHop();
        this.committedPayload = payload;
        this.pendingRound = null;
    }

    private void requireSession(SamplingClock clock) {
        if (clock.getSnapshotVersion() != snapshotVersion || clock.getSessionId() != sessionId) {
            throw new IllegalArgumentException("sampling clock does not match vertex session");
        }
    }

    public long getSnapshotVersion() {
        return snapshotVersion;
    }

    public long getSessionId() {
        return sessionId;
    }

    public int getCompletedHop() {
        return completedHop;
    }

    public P getCommittedPayload() {
        return committedPayload;
    }

    public PendingSamplingRound<K, EV> getPendingRound() {
        return pendingRound;
    }
}
