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

import java.lang.reflect.Array;
import java.util.List;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.types.ObjectType;

public final class GCNEdgeWeightExtractor {

    public static final double DEFAULT_EDGE_WEIGHT = 1.0D;

    private GCNEdgeWeightExtractor() {
    }

    public static double extract(Object edgeValue) {
        if (edgeValue == null) {
            return DEFAULT_EDGE_WEIGHT;
        }
        if (edgeValue instanceof Number) {
            return ((Number) edgeValue).doubleValue();
        }
        if (edgeValue instanceof Row) {
            return extractFromRow((Row) edgeValue);
        }
        if (edgeValue instanceof List) {
            List<?> values = (List<?>) edgeValue;
            return values.isEmpty() ? DEFAULT_EDGE_WEIGHT : extract(values.get(0));
        }
        if (edgeValue.getClass().isArray()) {
            return Array.getLength(edgeValue) == 0 ? DEFAULT_EDGE_WEIGHT
                : extract(Array.get(edgeValue, 0));
        }
        if (edgeValue instanceof CharSequence) {
            try {
                return Double.parseDouble(edgeValue.toString());
            } catch (NumberFormatException e) {
                return DEFAULT_EDGE_WEIGHT;
            }
        }
        return DEFAULT_EDGE_WEIGHT;
    }

    private static double extractFromRow(Row value) {
        try {
            return extract(value.getField(0, ObjectType.INSTANCE));
        } catch (Exception e) {
            return DEFAULT_EDGE_WEIGHT;
        }
    }
}
