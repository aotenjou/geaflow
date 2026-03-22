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
import org.apache.geaflow.common.type.IType;
import org.apache.geaflow.common.type.primitive.DoubleType;
import org.apache.geaflow.dsl.common.data.Row;
import org.apache.geaflow.dsl.common.data.RowVertex;
import org.apache.geaflow.dsl.common.data.impl.ObjectRow;
import org.apache.geaflow.dsl.common.types.GraphSchema;
import org.apache.geaflow.dsl.common.types.VertexType;

public class GCNFeatureCollector {

    public double[] collectFromRowVertex(RowVertex vertex, GraphSchema graphSchema, int featureDimLimit) {
        VertexType vertexType = graphSchema.getVertices().get(0);
        IType<?>[] valueTypes = vertexType.getValueTypes();
        double[] features = new double[featureDimLimit];
        for (int i = 0; i < featureDimLimit && i < valueTypes.length; i++) {
            Object value = vertex.getField(vertexType.getValueOffset() + i, valueTypes[i]);
            features[i] = toDouble(value);
        }
        return features;
    }

    public double[] collectFromValue(Object value, int featureDimLimit) {
        double[] features = new double[featureDimLimit];
        if (value == null) {
            return features;
        }
        if (value instanceof List) {
            List<?> list = (List<?>) value;
            for (int i = 0; i < featureDimLimit && i < list.size(); i++) {
                features[i] = toDouble(list.get(i));
            }
            return features;
        }
        if (value instanceof ObjectRow) {
            Object[] fields = ((ObjectRow) value).getFields();
            for (int i = 0; i < featureDimLimit && i < fields.length; i++) {
                features[i] = toDouble(fields[i]);
            }
            return features;
        }
        if (value instanceof Row) {
            for (int i = 0; i < featureDimLimit; i++) {
                try {
                    features[i] = toDouble(((Row) value).getField(i, DoubleType.INSTANCE));
                } catch (Exception e) {
                    break;
                }
            }
            return features;
        }
        if (value.getClass().isArray()) {
            int length = Array.getLength(value);
            for (int i = 0; i < featureDimLimit && i < length; i++) {
                features[i] = toDouble(Array.get(value, i));
            }
            return features;
        }
        features[0] = toDouble(value);
        return features;
    }

    private double toDouble(Object value) {
        if (value == null) {
            return 0D;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return Double.parseDouble(String.valueOf(value));
    }
}
