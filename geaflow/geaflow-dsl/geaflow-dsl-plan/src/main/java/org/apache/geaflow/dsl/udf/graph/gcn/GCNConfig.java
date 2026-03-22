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

package org.apache.geaflow.dsl.udf.graph.gcn;

import java.io.Serializable;

public class GCNConfig implements Serializable {

    public static final int DEFAULT_NUM_HOPS = 2;
    public static final int DEFAULT_NUM_SAMPLES_PER_HOP = 25;
    public static final boolean DEFAULT_WITH_SELF_LOOP = true;
    public static final int DEFAULT_FEATURE_DIM_LIMIT = 64;
    public static final long DEFAULT_RANDOM_SEED = 20260322L;
    public static final String DEFAULT_PYTHON_TRANSFORM_CLASS = "GCNTransFormFunction";

    private final int numHops;
    private final int numSamplesPerHop;
    private final String pythonTransformClassName;
    private final boolean withSelfLoop;
    private final int featureDimLimit;
    private final long randomSeed;

    public GCNConfig() {
        this(DEFAULT_NUM_HOPS, DEFAULT_NUM_SAMPLES_PER_HOP, DEFAULT_PYTHON_TRANSFORM_CLASS,
            DEFAULT_WITH_SELF_LOOP, DEFAULT_FEATURE_DIM_LIMIT, DEFAULT_RANDOM_SEED);
    }

    public GCNConfig(int numHops, int numSamplesPerHop, String pythonTransformClassName) {
        this(numHops, numSamplesPerHop, pythonTransformClassName, DEFAULT_WITH_SELF_LOOP,
            DEFAULT_FEATURE_DIM_LIMIT, DEFAULT_RANDOM_SEED);
    }

    public GCNConfig(int numHops, int numSamplesPerHop, String pythonTransformClassName,
                     boolean withSelfLoop, int featureDimLimit, long randomSeed) {
        if (numHops <= 0) {
            throw new IllegalArgumentException("numHops must be positive");
        }
        if (numSamplesPerHop <= 0) {
            throw new IllegalArgumentException("numSamplesPerHop must be positive");
        }
        if (featureDimLimit <= 0) {
            throw new IllegalArgumentException("featureDimLimit must be positive");
        }
        if (pythonTransformClassName == null || pythonTransformClassName.trim().isEmpty()) {
            throw new IllegalArgumentException("pythonTransformClassName cannot be blank");
        }
        this.numHops = numHops;
        this.numSamplesPerHop = numSamplesPerHop;
        this.pythonTransformClassName = pythonTransformClassName;
        this.withSelfLoop = withSelfLoop;
        this.featureDimLimit = featureDimLimit;
        this.randomSeed = randomSeed;
    }

    public int getNumHops() {
        return numHops;
    }

    public int getNumSamplesPerHop() {
        return numSamplesPerHop;
    }

    public String getPythonTransformClassName() {
        return pythonTransformClassName;
    }

    public boolean isWithSelfLoop() {
        return withSelfLoop;
    }

    public int getFeatureDimLimit() {
        return featureDimLimit;
    }

    public long getRandomSeed() {
        return randomSeed;
    }
}
