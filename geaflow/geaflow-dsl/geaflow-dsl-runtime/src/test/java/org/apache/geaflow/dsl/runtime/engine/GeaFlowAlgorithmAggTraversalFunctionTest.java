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

package org.apache.geaflow.dsl.runtime.engine;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import org.apache.geaflow.dsl.common.algo.AlgorithmUserFunction;
import org.apache.geaflow.dsl.common.types.GraphSchema;
import org.testng.annotations.Test;

public class GeaFlowAlgorithmAggTraversalFunctionTest {

    @Test
    public void testForwardsIterationLifecycle() {
        AlgorithmUserFunction<Object, Object> userFunction = mock(AlgorithmUserFunction.class);
        GeaFlowAlgorithmAggTraversalFunction function = new GeaFlowAlgorithmAggTraversalFunction(
            mock(GraphSchema.class), userFunction, new Object[0]);

        function.initIteration(3L);
        function.finishIteration(3L);

        verify(userFunction).initIteration(3L);
        verify(userFunction).finishIteration(3L);
    }
}
