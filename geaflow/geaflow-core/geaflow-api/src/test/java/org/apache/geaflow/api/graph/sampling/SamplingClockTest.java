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

import org.testng.Assert;
import org.testng.annotations.Test;

public class SamplingClockTest {

    @Test
    public void testMapsTwoHopsToFiveIterations() {
        SamplingClock firstRequest = SamplingClock.forIteration(7L, 11L, 2, 1L, 1L);
        SamplingClock firstResponse = SamplingClock.forIteration(7L, 11L, 2, 1L, 2L);
        SamplingClock firstCommit = SamplingClock.forIteration(7L, 11L, 2, 1L, 3L);
        SamplingClock secondResponse = SamplingClock.forIteration(7L, 11L, 2, 1L, 4L);
        SamplingClock complete = SamplingClock.forIteration(7L, 11L, 2, 1L, 5L);

        assertClock(firstRequest, 1, SamplingPhase.REQUEST);
        assertClock(firstResponse, 1, SamplingPhase.RESPOND);
        assertClock(firstCommit, 1, SamplingPhase.COMMIT_AND_REQUEST);
        assertClock(firstCommit.nextRequestClock(), 2, SamplingPhase.REQUEST);
        assertClock(secondResponse, 2, SamplingPhase.RESPOND);
        assertClock(complete, 2, SamplingPhase.COMPLETE);
        Assert.assertEquals(SamplingClock.requiredIterations(2), 5L);
    }

    @Test
    public void testSamplingVersionChangesPerHopButNotPhase() {
        SamplingClock request = new SamplingClock(7L, 11L, 1, SamplingPhase.REQUEST);
        SamplingClock response = request.responseClock();
        SamplingClock next = new SamplingClock(7L, 11L, 2, SamplingPhase.REQUEST);

        Assert.assertEquals(request.getSamplingVersion(), response.getSamplingVersion());
        Assert.assertNotEquals(request.getSamplingVersion(), next.getSamplingVersion());
        Assert.assertTrue(request.isSameRound(response));
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testRejectsIterationAfterSessionCompletion() {
        SamplingClock.forIteration(7L, 11L, 2, 1L, 6L);
    }

    private void assertClock(SamplingClock clock, int hop, SamplingPhase phase) {
        Assert.assertEquals(clock.getHop(), hop);
        Assert.assertEquals(clock.getPhase(), phase);
    }
}
