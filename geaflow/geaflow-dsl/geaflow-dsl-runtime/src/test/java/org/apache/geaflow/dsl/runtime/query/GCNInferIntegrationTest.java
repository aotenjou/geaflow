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

package org.apache.geaflow.dsl.runtime.query;

import com.google.common.io.Resources;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import org.apache.commons.io.FileUtils;
import org.apache.geaflow.common.config.Configuration;
import org.apache.geaflow.common.config.keys.ExecutionConfigKeys;
import org.apache.geaflow.common.config.keys.FrameworkConfigKeys;
import org.apache.geaflow.dsl.udf.graph.gcn.GCNInferPayload;
import org.apache.geaflow.file.FileConfigKeys;
import org.apache.geaflow.infer.InferContext;
import org.apache.geaflow.infer.InferEnvironmentManager;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

public class GCNInferIntegrationTest {

    private static final String TEST_WORK_DIR = "/tmp/geaflow/gcn_infer_test";
    private static final String PYTHON_UDF_DIR = TEST_WORK_DIR + "/python_udf";
    private static final String TEST_JOB_JAR = "gcn-test-job.jar";
    private static final String GCN_TRANSFORM_CLASS = "GCNTransFormFunction";
    private static final String FAILING_TRANSFORM_CLASS = "GCNFailingTransformFunction";
    private static final String CONDA_URL_ENV = "GEAFLOW_GCN_INFER_CONDA_URL";
    private static final String GCN_QUERY_PATH = "/query/gql_algorithm_gcn_infer_001.sql";

    @AfterMethod
    public void tearDown() {
        FileUtils.deleteQuietly(new File(TEST_WORK_DIR));
        FileUtils.deleteQuietly(new File(getClasspathRoot(), TEST_JOB_JAR));
    }

    @Test(timeOut = 30000)
    public void testGCNPythonUDFDirectWithoutModelFile() throws Exception {
        ensurePythonModuleAvailable("torch");
        File udfDir = new File(PYTHON_UDF_DIR);
        FileUtils.forceMkdir(udfDir);
        copyResourceToDirectory("TransFormFunctionUDF.py", udfDir);

        String testScript = String.join("\n",
            "import os",
            "import sys",
            "os.chdir('" + escapeForPython(udfDir.getAbsolutePath()) + "')",
            "sys.path.insert(0, os.getcwd())",
            "from TransFormFunctionUDF import GCNTransFormFunction",
            "payload = {",
            "  'center_node_id': 1,",
            "  'sampled_nodes': [1, 2, 3],",
            "  'node_features': [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],",
            "  'edge_index': [[0, 0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 0, 1, 2]]",
            "}",
            "func = GCNTransFormFunction()",
            "assert func.model_loaded is False",
            "result, center = func.transform_pre(payload)",
            "output = func.transform_post(result)",
            "assert center == 1",
            "assert output['node_id'] == 1",
            "assert len(output['embedding']) == 16",
            "assert 'predicted_class' in output",
            "assert 'confidence' in output",
            "assert 0.0 <= float(output['confidence']) <= 1.0",
            "print('GCN direct python test passed')"
        );

        File scriptFile = new File(udfDir, "test_gcn_udf.py");
        try (OutputStreamWriter writer = new OutputStreamWriter(
            new FileOutputStream(scriptFile), StandardCharsets.UTF_8)) {
            writer.write(testScript);
        }

        Process process = new ProcessBuilder(findPythonExecutable(), scriptFile.getAbsolutePath())
            .directory(udfDir)
            .redirectErrorStream(true)
            .start();
        String output = readProcessOutput(process);
        int exitCode = process.waitFor();

        Assert.assertEquals(exitCode, 0, output);
        Assert.assertTrue(output.contains("GCN direct python test passed"), output);
    }

    @Test
    public void testPythonModulesAvailable() throws Exception {
        ensurePythonModuleAvailable("torch");
    }

    @Test(timeOut = 600000)
    public void testGCNInferContextEndToEndWhenCondaConfigured() throws Exception {
        String condaUrl = System.getenv(CONDA_URL_ENV);
        if (condaUrl == null || condaUrl.trim().isEmpty()) {
            throw new SkipException("Skip GCN infer end-to-end test: " + CONDA_URL_ENV + " is not set");
        }

        ensureInferEnvironmentReady(condaUrl);

        InferContext<Object> inferContext = new InferContext<>(
            createInferConfiguration(condaUrl, GCN_TRANSFORM_CLASS));
        try {
            assertInferResult(inferContext.infer(createPayload(1L)), 1L);
        } finally {
            inferContext.close();
        }
    }

    @Test(timeOut = 600000)
    public void testGCNInferContextMultipleCallsWhenCondaConfigured() throws Exception {
        String condaUrl = System.getenv(CONDA_URL_ENV);
        if (condaUrl == null || condaUrl.trim().isEmpty()) {
            throw new SkipException("Skip repeated GCN infer test: " + CONDA_URL_ENV + " is not set");
        }

        ensureInferEnvironmentReady(condaUrl);

        InferContext<Object> inferContext = new InferContext<>(
            createInferConfiguration(condaUrl, GCN_TRANSFORM_CLASS));
        try {
            assertInferResult(inferContext.infer(createPayload(1L)), 1L);
            assertInferResult(inferContext.infer(createPayload(2L)), 2L);
            assertInferResult(inferContext.infer(createPayload(3L)), 3L);
        } finally {
            inferContext.close();
        }
    }

    @Test(timeOut = 600000)
    public void testGCNInferContextPropagatesPythonExceptionWhenCondaConfigured() throws Exception {
        String condaUrl = System.getenv(CONDA_URL_ENV);
        if (condaUrl == null || condaUrl.trim().isEmpty()) {
            throw new SkipException("Skip GCN python exception test: " + CONDA_URL_ENV + " is not set");
        }

        ensureInferEnvironmentReady(condaUrl);

        InferContext<Object> inferContext = new InferContext<>(
            createInferConfiguration(condaUrl, FAILING_TRANSFORM_CLASS));
        try {
            Object rawResult = inferContext.infer(createPayload(1L));
            Assert.assertTrue(rawResult instanceof String,
                "Expected python exception string, but got: " + rawResult);
            String error = (String) rawResult;
            Assert.assertTrue(error.startsWith("python_exception:"),
                "Expected python exception marker, but got: " + error);
            Assert.assertTrue(error.contains("gcn failing transform invoked"), error);
        } finally {
            inferContext.close();
        }
    }

    @Test(timeOut = 600000)
    public void testGCNQueryRuntimeEndToEndWhenCondaConfigured() throws Exception {
        String condaUrl = System.getenv(CONDA_URL_ENV);
        if (condaUrl == null || condaUrl.trim().isEmpty()) {
            throw new SkipException("Skip GCN DSL infer query test: " + CONDA_URL_ENV + " is not set");
        }

        createTestJobJar();

        QueryTester
            .build()
            .withConfig(FrameworkConfigKeys.INFER_ENV_ENABLE.getKey(), "true")
            .withConfig(FrameworkConfigKeys.INFER_ENV_CONDA_URL.getKey(), condaUrl)
            .withConfig(FrameworkConfigKeys.INFER_ENV_USER_TRANSFORM_CLASSNAME.getKey(),
                GCN_TRANSFORM_CLASS)
            .withConfig(FrameworkConfigKeys.INFER_ENV_INIT_TIMEOUT_SEC.getKey(), "300")
            .withConfig(ExecutionConfigKeys.JOB_WORK_PATH.getKey(), TEST_WORK_DIR)
            .withConfig(FileConfigKeys.USER_NAME.getKey(), "gcn_test_user")
            .withQueryPath(GCN_QUERY_PATH)
            .execute()
            .checkSinkResult();
    }

    private void ensurePythonModuleAvailable(String moduleName) throws Exception {
        String pythonExecutable = findPythonExecutable();
        Process process = new ProcessBuilder(
            pythonExecutable, "-c", "import " + moduleName + "; print('ok')").start();
        String output = readProcessOutput(process);
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new SkipException("Skip python integration test, missing module " + moduleName
                + ": " + output);
        }
    }

    private String findPythonExecutable() {
        String[] candidates = new String[]{"python3", "python"};
        for (String candidate : candidates) {
            try {
                Process process = new ProcessBuilder(candidate, "--version")
                    .redirectErrorStream(true)
                    .start();
                int exitCode = process.waitFor();
                if (exitCode == 0) {
                    return candidate;
                }
            } catch (Exception e) {
                // try next candidate
            }
        }
        throw new SkipException("Skip python integration test, no python executable found");
    }

    private String readProcessOutput(Process process) throws IOException {
        StringBuilder builder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line).append('\n');
            }
        }
        return builder.toString();
    }

    private void copyResourceToDirectory(String resourceName, File targetDirectory) throws IOException {
        try (InputStream inputStream = getRequiredResource(resourceName).openStream()) {
            File targetFile = new File(targetDirectory, resourceName);
            FileUtils.copyInputStreamToFile(inputStream, targetFile);
        }
    }

    private URL getRequiredResource(String resourceName) {
        URL resource = GCNInferIntegrationTest.class.getClassLoader().getResource(resourceName);
        Assert.assertNotNull(resource, "Missing resource " + resourceName);
        return resource;
    }

    private void createTestJobJar() throws IOException {
        File classpathRoot = getClasspathRoot();
        File jarFile = new File(classpathRoot, TEST_JOB_JAR);
        if (jarFile.exists()) {
            FileUtils.forceDelete(jarFile);
        }
        try (JarOutputStream jarOutputStream = new JarOutputStream(new FileOutputStream(jarFile))) {
            writeStringEntry(jarOutputStream, "TransFormFunctionUDF.py", buildTestPythonUdfContent());
            writeResourceEntry(jarOutputStream, "requirements.txt");
        }
    }

    private void ensureInferEnvironmentReady(String condaUrl) throws Exception {
        createTestJobJar();
        InferEnvironmentManager.buildInferEnvironmentManager(
            createInferConfiguration(condaUrl, GCN_TRANSFORM_CLASS)).createEnvironment();
    }

    private Configuration createInferConfiguration(String condaUrl, String transformClass) {
        Configuration config = new Configuration();
        config.put(FrameworkConfigKeys.INFER_ENV_ENABLE.getKey(), "true");
        config.put(FrameworkConfigKeys.INFER_ENV_CONDA_URL.getKey(), condaUrl);
        config.put(FrameworkConfigKeys.INFER_ENV_USER_TRANSFORM_CLASSNAME.getKey(), transformClass);
        config.put(FrameworkConfigKeys.INFER_ENV_INIT_TIMEOUT_SEC.getKey(), "300");
        config.put(ExecutionConfigKeys.JOB_UNIQUE_ID.getKey(),
            "gcn-infer-test-" + System.currentTimeMillis());
        config.put(ExecutionConfigKeys.JOB_WORK_PATH.getKey(), TEST_WORK_DIR);
        config.put(FileConfigKeys.USER_NAME.getKey(), "gcn_test_user");
        return config;
    }

    private void assertInferResult(Object rawResult, long expectedNodeId) {
        Assert.assertTrue(rawResult instanceof Map,
            "Expected Python dict result, but got: " + rawResult);
        Map<?, ?> result = (Map<?, ?>) rawResult;
        Assert.assertEquals(result.get("node_id"), expectedNodeId);
        Assert.assertNotNull(result.get("embedding"));
        Assert.assertNotNull(result.get("predicted_class"));
        Assert.assertNotNull(result.get("confidence"));
    }

    private GCNInferPayload createPayload(long centerNodeId) {
        if (centerNodeId == 1L) {
            return new GCNInferPayload(
                1L,
                Arrays.<Object>asList(1L, 2L, 3L),
                Arrays.asList(
                    featureVector(1.0D, 2.0D),
                    featureVector(0.0D, 1.0D),
                    featureVector(2.0D, 1.0D)
                ),
                new int[][]{{0, 0, 0, 1, 2}, {1, 2, 0, 1, 2}},
                null
            );
        }
        if (centerNodeId == 2L) {
            return new GCNInferPayload(
                2L,
                Arrays.<Object>asList(2L, 1L),
                Arrays.asList(
                    featureVector(0.0D, 1.0D),
                    featureVector(1.0D, 2.0D)
                ),
                new int[][]{{0, 0, 1}, {1, 0, 1}},
                null
            );
        }
        if (centerNodeId == 3L) {
            return new GCNInferPayload(
                3L,
                Arrays.<Object>asList(3L, 1L),
                Arrays.asList(
                    featureVector(2.0D, 1.0D),
                    featureVector(1.0D, 2.0D)
                ),
                new int[][]{{0, 0, 1}, {1, 0, 1}},
                null
            );
        }
        throw new IllegalArgumentException("Unsupported center node id " + centerNodeId);
    }

    private double[] featureVector(double first, double second) {
        double[] values = new double[64];
        values[0] = first;
        values[1] = second;
        return values;
    }

    private void writeResourceEntry(JarOutputStream jarOutputStream, String resourceName)
        throws IOException {
        JarEntry entry = new JarEntry(resourceName);
        jarOutputStream.putNextEntry(entry);
        try (InputStream inputStream = getRequiredResource(resourceName).openStream()) {
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                jarOutputStream.write(buffer, 0, bytesRead);
            }
        }
        jarOutputStream.closeEntry();
    }

    private void writeStringEntry(JarOutputStream jarOutputStream, String resourceName, String content)
        throws IOException {
        JarEntry entry = new JarEntry(resourceName);
        jarOutputStream.putNextEntry(entry);
        jarOutputStream.write(content.getBytes(StandardCharsets.UTF_8));
        jarOutputStream.closeEntry();
    }

    private String buildTestPythonUdfContent() throws IOException {
        StringBuilder builder = new StringBuilder();
        try (InputStream inputStream = getRequiredResource("TransFormFunctionUDF.py").openStream();
             InputStreamReader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);
             BufferedReader bufferedReader = new BufferedReader(reader)) {
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                builder.append(line).append('\n');
            }
        }
        builder.append('\n');
        builder.append("class ").append(FAILING_TRANSFORM_CLASS).append("(object):\n");
        builder.append("    input_size = 1\n\n");
        builder.append("    def transform_pre(self, *inputs):\n");
        builder.append("        raise RuntimeError('gcn failing transform invoked')\n\n");
        builder.append("    def transform_post(self, *inputs):\n");
        builder.append("        return None\n");
        return builder.toString();
    }

    private File getClasspathRoot() {
        try {
            return Paths.get(Resources.getResource(".").toURI()).toFile();
        } catch (Exception e) {
            throw new RuntimeException("Failed to locate classpath root", e);
        }
    }

    private String escapeForPython(String path) {
        return path.replace("\\", "\\\\").replace("'", "\\'");
    }
}
