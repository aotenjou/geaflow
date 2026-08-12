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

package org.apache.geaflow.state.sampling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.geaflow.common.iterator.CloseableIterator;
import org.apache.geaflow.model.graph.edge.EdgeDirection;
import org.apache.geaflow.model.graph.edge.IEdge;
import org.apache.geaflow.model.graph.edge.impl.ValueEdge;
import org.apache.geaflow.model.graph.vertex.impl.ValueVertex;
import org.apache.geaflow.state.data.OneDegreeGraph;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DeterministicNeighborSamplerTest {

    @Test
    public void testSamplingIsBoundedAndIndependentOfInputOrder() {
        List<IEdge<Long, String>> first = Arrays.asList(
            edge(1L, 2L), edge(1L, 3L), edge(1L, 4L));
        List<IEdge<Long, String>> second = Arrays.asList(
            edge(1L, 4L), edge(1L, 2L), edge(1L, 3L));

        List<IEdge<Long, String>> sampledFirst =
            DeterministicNeighborSampler.sample(1L, first, EdgeDirection.OUT, 2);
        List<IEdge<Long, String>> sampledSecond =
            DeterministicNeighborSampler.sample(1L, second, EdgeDirection.OUT, 2);

        Assert.assertEquals(sampledFirst.size(), 2);
        Assert.assertEquals(targetIds(sampledFirst), targetIds(sampledSecond));
    }

    @Test
    public void testPositiveFanoutDoesNotMaterializeAllCandidates() {
        List<IEdge<Long, String>> edges = new ArrayList<>();
        for (long target = 2L; target < 102L; target++) {
            edges.add(edge(1L, target));
        }

        List<IEdge<Long, String>> sampled = DeterministicNeighborSampler.sample(
            1L, edges, EdgeDirection.OUT, 3, Long::compare, 3L, 17L, 9L);

        Assert.assertEquals(sampled.stream().map(IEdge::getTargetId).distinct().count(), 3L);
        Assert.assertEquals(sampled.size(), 3);
    }

    @Test
    public void testSeedAndVersionAreStableAndInputOrderIndependent() {
        List<IEdge<Long, String>> first = new ArrayList<>();
        for (long target = 2L; target < 42L; target++) {
            first.add(edge(1L, target));
        }
        List<IEdge<Long, String>> reversed = new ArrayList<>(first);
        java.util.Collections.reverse(reversed);

        List<Long> firstSample = targetIds(DeterministicNeighborSampler.sample(
            1L, first, EdgeDirection.OUT, 5, Long::compare, 5L, 123L, 7L));
        List<Long> reorderedSample = targetIds(DeterministicNeighborSampler.sample(
            1L, reversed, EdgeDirection.OUT, 5, Long::compare, 5L, 123L, 7L));
        List<Long> nextVersionSample = targetIds(DeterministicNeighborSampler.sample(
            1L, first, EdgeDirection.OUT, 5, Long::compare, 5L, 123L, 8L));

        Assert.assertEquals(firstSample, reorderedSample);
        Assert.assertNotEquals(firstSample, nextVersionSample);
    }

    @Test
    public void testDirectionAndUnlimitedFanout() {
        IEdge<Long, String> out = edge(1L, 2L);
        IEdge<Long, String> in = edge(1L, 3L);
        in.setDirect(EdgeDirection.IN);

        List<IEdge<Long, String>> sampledIn = DeterministicNeighborSampler.sample(
            1L, Arrays.asList(out, in), EdgeDirection.IN, -1);
        Assert.assertEquals(sampledIn.size(), 1);
        Assert.assertEquals(sampledIn.get(0).getSrcId(), Long.valueOf(3L));
        Assert.assertEquals(sampledIn.get(0).getTargetId(), Long.valueOf(1L));
        Assert.assertEquals(sampledIn.get(0).getDirect(), EdgeDirection.IN);
        Assert.assertEquals(
            DeterministicNeighborSampler.sample(1L, Arrays.asList(out, in), EdgeDirection.BOTH, -1).size(),
            2);
    }

    @Test
    public void testFanoutCountsNeighborsAndKeepsSelectedParallelEdges() {
        List<IEdge<Long, String>> edges = Arrays.asList(
            edge(1L, 2L), edgeWithValue(1L, 2L, "parallel"), edge(1L, 3L), edge(1L, 4L));

        List<IEdge<Long, String>> sampled = DeterministicNeighborSampler.sample(
            1L, edges, EdgeDirection.OUT, 2);
        Map<Long, Long> allCounts = edges.stream().collect(Collectors.groupingBy(
            IEdge::getTargetId, Collectors.counting()));
        Map<Long, Long> sampledCounts = sampled.stream().collect(Collectors.groupingBy(
            IEdge::getTargetId, Collectors.counting()));

        Assert.assertEquals(sampledCounts.size(), 2);
        for (Map.Entry<Long, Long> entry : sampledCounts.entrySet()) {
            Assert.assertEquals(entry.getValue(), allCounts.get(entry.getKey()));
        }
    }

    @Test
    public void testOneDegreeStateExposesOneHopSampling() {
        TrackingIterator iterator = new TrackingIterator(Arrays.asList(
            edge(1L, 2L), edge(1L, 3L), edge(1L, 4L)).iterator());
        OneDegreeGraph<Long, String, String> oneDegreeGraph = new OneDegreeGraph<>(1L,
            new ValueVertex<>(1L, "vertex"), iterator);

        List<IEdge<Long, String>> sampled = oneDegreeGraph.sampleNeighbors(
            EdgeDirection.OUT, 2);

        Assert.assertEquals(sampled.size(), 2);
        Assert.assertTrue(iterator.closed);
        Assert.assertEquals(oneDegreeGraph.sampleNeighbors(EdgeDirection.OUT, 2), sampled);
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testOneDegreeStateRejectsDifferentRequestAfterIteratorConsumption() {
        OneDegreeGraph<Long, String, String> oneDegreeGraph = new OneDegreeGraph<>(1L,
            new ValueVertex<>(1L, "vertex"), new TrackingIterator(Arrays.asList(
                edge(1L, 2L), edge(1L, 3L), edge(1L, 4L)).iterator()));

        oneDegreeGraph.sampleNeighbors(EdgeDirection.OUT, 2);
        oneDegreeGraph.sampleNeighbors(EdgeDirection.OUT, 1);
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testOneDegreeStateRejectsDifferentSamplingVersion() {
        OneDegreeGraph<Long, String, String> oneDegreeGraph = new OneDegreeGraph<>(1L,
            new ValueVertex<>(1L, "vertex"), new TrackingIterator(Arrays.asList(
                edge(1L, 2L), edge(1L, 3L), edge(1L, 4L)).iterator()));

        oneDegreeGraph.sampleNeighbors(EdgeDirection.OUT, 2, 10L, 17L, 1L);
        oneDegreeGraph.sampleNeighbors(EdgeDirection.OUT, 2, 10L, 17L, 2L);
    }

    @Test
    public void testNeighborhoodMatchesSnapshotAndSamplingVersion() {
        LocalNeighborhood<Long, String, String> neighborhood = new LocalNeighborhood<>(
            new ValueVertex<>(1L, "vertex"), Arrays.asList(edge(1L, 2L)), 7L, 3L);

        Assert.assertTrue(neighborhood.matches(7L, 3L));
        Assert.assertFalse(neighborhood.matches(8L, 3L));
        Assert.assertFalse(neighborhood.matches(7L, 4L));
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void testRejectsCandidateEdgeOverflow() {
        DeterministicNeighborSampler.sample(1L,
            Arrays.asList(edge(1L, 2L), edge(1L, 3L)), EdgeDirection.OUT, -1,
            Long::compare, 1);
    }

    private static IEdge<Long, String> edge(long source, long target) {
        return edgeWithValue(source, target, "value");
    }

    private static IEdge<Long, String> edgeWithValue(long source, long target, String value) {
        ValueEdge<Long, String> edge = new ValueEdge<>(source, target, value);
        edge.setDirect(EdgeDirection.OUT);
        return edge;
    }

    private static List<Long> targetIds(List<IEdge<Long, String>> edges) {
        return edges.stream().map(IEdge::getTargetId).collect(Collectors.toList());
    }

    private static class TrackingIterator implements CloseableIterator<IEdge<Long, String>> {

        private final Iterator<IEdge<Long, String>> delegate;
        private boolean closed;

        private TrackingIterator(Iterator<IEdge<Long, String>> delegate) {
            this.delegate = delegate;
        }

        @Override
        public void close() {
            closed = true;
        }

        @Override
        public boolean hasNext() {
            return delegate.hasNext();
        }

        @Override
        public IEdge<Long, String> next() {
            return delegate.next();
        }
    }
}
