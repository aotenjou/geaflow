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

package org.apache.geaflow.state.data;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import org.apache.geaflow.common.iterator.CloseableIterator;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.vertex.IVertex;
import org.apache.geaflow.state.sampling.DeterministicNeighborSampler;

public class OneDegreeGraph<K, VV, EV> implements Serializable {

    private IVertex<K, VV> vertex;
    protected CloseableIterator<IEdge<K, EV>> edgeIterator;
    protected K key;
    private List<IEdge<K, EV>> sampledEdges;
    private EdgeDirection sampledDirection;
    private Integer sampledFanout;
    private Long sampledMaxReturnedEdges;
    private Long sampledSeed;
    private Long sampledVersion;

    public OneDegreeGraph(K key, IVertex<K, VV> vertex, CloseableIterator<IEdge<K, EV>> edgeIterator) {
        this.key = key;
        this.vertex = vertex;
        this.edgeIterator = edgeIterator;
    }

    public K getKey() {
        return key;
    }

    public IVertex<K, VV> getVertex() {
        return vertex;
    }

    public CloseableIterator<IEdge<K, EV>> getEdgeIterator() {
        return edgeIterator;
    }

    /** Samples this vertex's bounded one-hop neighborhood in the state layer. */
    public synchronized List<IEdge<K, EV>> sampleNeighbors(EdgeDirection direction, int fanout) {
        return sampleNeighbors(direction, fanout,
            DeterministicNeighborSampler.DEFAULT_MAX_CANDIDATE_EDGES, 0L, 0L);
    }

    /** Samples this one-shot edge iterator for one deterministic sampling round. */
    public synchronized List<IEdge<K, EV>> sampleNeighbors(EdgeDirection direction, int fanout,
                                                            long maxReturnedEdges, long seed,
                                                            long samplingVersion) {
        if (sampledEdges != null) {
            if (sampledDirection != direction || !Objects.equals(sampledFanout, fanout)
                || !Objects.equals(sampledMaxReturnedEdges, maxReturnedEdges)
                || !Objects.equals(sampledSeed, seed)
                || !Objects.equals(sampledVersion, samplingVersion)) {
                throw new IllegalStateException(
                    "one-degree edge iterator was already consumed by a different sampling request");
            }
            return sampledEdges;
        }
        try {
            Iterable<IEdge<K, EV>> iterable = () -> edgeIterator;
            sampledEdges = Collections.unmodifiableList(
                DeterministicNeighborSampler.sample(key, iterable, direction, fanout,
                    java.util.Comparator.comparing(String::valueOf), maxReturnedEdges,
                    seed, samplingVersion));
            sampledDirection = direction;
            sampledFanout = fanout;
            sampledMaxReturnedEdges = maxReturnedEdges;
            sampledSeed = seed;
            sampledVersion = samplingVersion;
            return sampledEdges;
        } finally {
            edgeIterator.close();
        }
    }
}
