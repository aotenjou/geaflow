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

package org.apache.geaflow.common.config;

import static org.apache.geaflow.common.config.keys.FrameworkConfigKeys.INFER_ENV_USER_TRANSFORM_CLASSNAME;

import java.util.HashMap;
import java.util.Map;

public final class InferConfigHelper {

    private InferConfigHelper() {
    }

    public static boolean hasTransformClassName(Configuration configuration) {
        return getTransformClassName(configuration) != null;
    }

    public static String getTransformClassName(Configuration configuration) {
        if (configuration == null || !configuration.contains(INFER_ENV_USER_TRANSFORM_CLASSNAME)) {
            return null;
        }
        String transformClassName = configuration.getString(
            INFER_ENV_USER_TRANSFORM_CLASSNAME.getKey(), null);
        if (transformClassName == null) {
            return null;
        }
        String normalizedClassName = transformClassName.trim();
        return normalizedClassName.isEmpty() ? null : normalizedClassName;
    }

    public static Configuration buildInferConfiguration(Configuration baseConfig,
                                                        String pythonTransformClassName) {
        if (baseConfig == null || pythonTransformClassName == null) {
            return baseConfig;
        }
        String normalizedClassName = pythonTransformClassName.trim();
        if (normalizedClassName.isEmpty()) {
            return baseConfig;
        }
        if (normalizedClassName.equals(getTransformClassName(baseConfig))) {
            return baseConfig;
        }
        Map<String, String> configMap = new HashMap<>(baseConfig.getConfigMap());
        configMap.put(INFER_ENV_USER_TRANSFORM_CLASSNAME.getKey(), normalizedClassName);
        Configuration configuration = new Configuration(configMap);
        configuration.setMasterId(baseConfig.getMasterId());
        return configuration;
    }
}
