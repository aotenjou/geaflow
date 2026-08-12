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

package org.apache.geaflow.dsl.common.algo;

import org.apache.geaflow.api.graph.sampling.SamplingClock;
import org.apache.geaflow.api.graph.sampling.SamplingPhase;
import org.apache.geaflow.api.graph.sampling.SubgraphSamplingSpec;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.data.RowVertex;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.state.sampling.LocalNeighborhood;

/** Runtime-facing contract for reusable one-hop sampling. */
public interface AlgorithmSamplingRuntimeContext<K, M> extends AlgorithmRuntimeContext<K, M> {

    LocalNeighborhood<Object, Row, Row> sampleOneHop(RowVertex vertex, EdgeDirection direction,
                                                     int fanout);

    default LocalNeighborhood<Object, Row, Row> sampleOneHop(RowVertex vertex,
                                                              EdgeDirection direction,
                                                              int fanout,
                                                              long maxCandidateEdges) {
        LocalNeighborhood<Object, Row, Row> neighborhood = sampleOneHop(vertex, direction, fanout);
        if (neighborhood.getEdges().size() > maxCandidateEdges) {
            throw new IllegalStateException(String.format(
                "one-hop sampling edge limit exceeded, vertexId=%s, actual=%s, limit=%s",
                vertex.getId(), neighborhood.getEdges().size(), maxCandidateEdges));
        }
        return neighborhood;
    }

    default LocalNeighborhood<Object, Row, Row> sampleOneHop(RowVertex vertex,
                                                              EdgeDirection direction,
                                                              int fanout,
                                                              long maxReturnedEdges,
                                                              long seed,
                                                              long samplingVersion) {
        return sampleOneHop(vertex, direction, fanout, maxReturnedEdges);
    }

    default SamplingClock getSamplingClock(SubgraphSamplingSpec spec, long sessionId,
                                           long startIterationId) {
        return SamplingClock.forIteration(getSamplingSnapshotVersion(), sessionId,
            spec.getHops(), startIterationId, getCurrentIterationId());
    }

    default LocalNeighborhood<Object, Row, Row> sampleOneHop(RowVertex vertex,
                                                              SubgraphSamplingSpec spec,
                                                              SamplingClock requestClock) {
        if (requestClock.getPhase() != SamplingPhase.REQUEST) {
            throw new IllegalArgumentException("one-hop sampling requires a request clock");
        }
        if (requestClock.getSnapshotVersion() != getSamplingSnapshotVersion()) {
            throw new IllegalArgumentException("sampling clock does not match runtime snapshot");
        }
        return sampleOneHop(vertex, spec.getDirection(), spec.getFanout(),
            spec.getMaxReturnedEdges(), spec.getSeed(), requestClock.getSamplingVersion());
    }

    long getSamplingSnapshotVersion();

    default long getNeighborhoodChangeVersion(Object vertexId) {
        return Long.MIN_VALUE;
    }
}
