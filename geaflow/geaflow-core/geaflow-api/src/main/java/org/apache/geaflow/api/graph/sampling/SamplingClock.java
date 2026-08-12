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

/** Immutable logical clock mapped from the runtime BSP iteration. */
public final class SamplingClock implements Serializable {

    private final long snapshotVersion;
    private final long sessionId;
    private final int hop;
    private final SamplingPhase phase;

    public SamplingClock(long snapshotVersion, long sessionId, int hop, SamplingPhase phase) {
        if (hop < 1) {
            throw new IllegalArgumentException("sampling hop must be greater than zero");
        }
        this.snapshotVersion = snapshotVersion;
        this.sessionId = sessionId;
        this.hop = hop;
        this.phase = Objects.requireNonNull(phase, "phase");
    }

    public static SamplingClock forIteration(long snapshotVersion, long sessionId, int maxHops,
                                             long startIterationId, long iterationId) {
        if (maxHops < 1) {
            throw new IllegalArgumentException("maxHops must be greater than zero");
        }
        if (iterationId < startIterationId) {
            throw new IllegalArgumentException("iteration precedes the sampling session");
        }
        long offset = iterationId - startIterationId;
        if (offset == 0L) {
            return new SamplingClock(snapshotVersion, sessionId, 1, SamplingPhase.REQUEST);
        }
        if ((offset & 1L) == 1L) {
            long responseHop = (offset + 1L) / 2L;
            requireHopInRange(responseHop, maxHops, iterationId);
            return new SamplingClock(snapshotVersion, sessionId, (int) responseHop,
                SamplingPhase.RESPOND);
        }
        long completedHop = offset / 2L;
        requireHopInRange(completedHop, maxHops, iterationId);
        SamplingPhase phase = completedHop == maxHops
            ? SamplingPhase.COMPLETE : SamplingPhase.COMMIT_AND_REQUEST;
        return new SamplingClock(snapshotVersion, sessionId, (int) completedHop, phase);
    }

    public static long requiredIterations(int maxHops) {
        if (maxHops < 1) {
            throw new IllegalArgumentException("maxHops must be greater than zero");
        }
        return Math.addExact(Math.multiplyExact((long) maxHops, 2L), 1L);
    }

    private static void requireHopInRange(long hop, int maxHops, long iterationId) {
        if (hop < 1L || hop > maxHops) {
            throw new IllegalArgumentException("iteration is outside the sampling session: "
                + iterationId);
        }
    }

    public SamplingClock responseClock() {
        if (phase != SamplingPhase.REQUEST) {
            throw new IllegalStateException("only a request clock can create a response clock");
        }
        return new SamplingClock(snapshotVersion, sessionId, hop, SamplingPhase.RESPOND);
    }

    public SamplingClock nextRequestClock() {
        if (phase != SamplingPhase.COMMIT_AND_REQUEST) {
            throw new IllegalStateException("current clock does not start another sampling hop");
        }
        return new SamplingClock(snapshotVersion, sessionId, Math.addExact(hop, 1),
            SamplingPhase.REQUEST);
    }

    public boolean isSameRound(SamplingClock other) {
        return other != null && snapshotVersion == other.snapshotVersion
            && sessionId == other.sessionId && hop == other.hop;
    }

    public long getSamplingVersion() {
        long value = mix64(snapshotVersion) ^ Long.rotateLeft(mix64(sessionId), 21);
        return mix64(value ^ Long.rotateLeft(mix64(hop), 42));
    }

    private static long mix64(long value) {
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }

    public long getSnapshotVersion() {
        return snapshotVersion;
    }

    public long getSessionId() {
        return sessionId;
    }

    public int getHop() {
        return hop;
    }

    public SamplingPhase getPhase() {
        return phase;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof SamplingClock)) {
            return false;
        }
        SamplingClock that = (SamplingClock) other;
        return snapshotVersion == that.snapshotVersion && sessionId == that.sessionId
            && hop == that.hop && phase == that.phase;
    }

    @Override
    public int hashCode() {
        return Objects.hash(snapshotVersion, sessionId, hop, phase);
    }
}
