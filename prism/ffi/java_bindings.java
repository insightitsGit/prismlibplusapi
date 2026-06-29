package io.insightits.prism.driver;

import java.util.Arrays;

/**
 * PrismDriver Java JNI bindings.
 *
 * <p>The native library (prism_driver_jni.so / prism_driver_jni.dll) wraps the
 * core prism_driver C ABI with JNI glue code that handles type conversions.
 *
 * <p>Build the JNI wrapper:
 * <pre>
 *   javac -h prism/ffi/jni_src prism/ffi/java_bindings.java
 *   # Then compile prism/ffi/jni_src/io_insightits_prism_driver_PrismDriver.cpp
 *   # against prism_driver.h and link with -lprism_driver
 * </pre>
 *
 * <p>Usage:
 * <pre>
 *   PrismDriver driver = new PrismDriver("db-proxy-1", 50051, "acme", null);
 *   driver.connect();
 *
 *   PrismDriver.QueryResult[] results = driver.query("orders", embedding, 10, 0.8f);
 *   for (var r : results)
 *       System.out.printf("%.4f  %s%n", r.score, r.textRepr);
 *
 *   driver.write("orders", embedding, "order #42");
 *   driver.disconnect();
 * </pre>
 *
 * <p>Maven dependency (after publishing):
 * <pre>
 *   &lt;dependency&gt;
 *     &lt;groupId&gt;io.insightits&lt;/groupId&gt;
 *     &lt;artifactId&gt;prism-driver&lt;/artifactId&gt;
 *     &lt;version&gt;0.2.0&lt;/version&gt;
 *   &lt;/dependency&gt;
 * </pre>
 */
public final class PrismDriver implements AutoCloseable {

    // ── Native library loading ───────────────────────────────────────────────
    static {
        String libPath = System.getenv("PRISM_DRIVER_LIB");
        if (libPath != null) {
            System.load(libPath);
        } else {
            System.loadLibrary("prism_driver_jni");
        }
    }

    // ── Data types ───────────────────────────────────────────────────────────

    public static final class QueryResult {
        public final String  eventId;
        public final String  rowId;
        public final float   score;
        public final String  textRepr;
        public final float[] vector;

        public QueryResult(
                String eventId, String rowId,
                float score, String textRepr, float[] vector) {
            this.eventId  = eventId;
            this.rowId    = rowId;
            this.score    = score;
            this.textRepr = textRepr;
            this.vector   = vector != null ? Arrays.copyOf(vector, vector.length)
                                           : new float[0];
        }

        @Override
        public String toString() {
            return String.format("QueryResult{rowId='%s', score=%.4f, text='%s'}",
                    rowId, score, textRepr);
        }
    }

    public static final class PrismDriverException extends RuntimeException {
        public PrismDriverException(String message) { super(message); }
    }

    // ── JNI native methods ───────────────────────────────────────────────────
    // The C implementation lives in prism/ffi/jni_src/

    private static native long  nativeConnect(
            String host, int port, String tenantId, String tlsCert);
    private static native void  nativeDisconnect(long handle);
    private static native QueryResult[] nativeQuery(
            long handle, String table, float[] vector, int topK, float threshold);
    private static native void  nativeWrite(
            long handle, String table, float[] vector, String textRepr);
    private static native String nativeLastError(long handle);
    public  static native String version();

    // ── Fields ───────────────────────────────────────────────────────────────

    private final String host;
    private final int    port;
    private final String tenantId;
    private final String tlsCert;
    private long   handle    = 0L;
    private boolean connected = false;

    // ── Constructor ──────────────────────────────────────────────────────────

    public PrismDriver(String host, int port, String tenantId, String tlsCert) {
        this.host     = host;
        this.port     = port;
        this.tenantId = tenantId != null ? tenantId : "";
        this.tlsCert  = tlsCert;
    }

    public PrismDriver(String host) {
        this(host, 50051, "", null);
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    public synchronized void connect() {
        handle = nativeConnect(host, port, tenantId, tlsCert);
        if (handle == 0L) {
            throw new PrismDriverException(
                    "prism_connect failed: " + nativeLastError(0L));
        }
        connected = true;
    }

    public synchronized void disconnect() {
        if (connected && handle != 0L) {
            nativeDisconnect(handle);
            handle    = 0L;
            connected = false;
        }
    }

    @Override
    public void close() {
        disconnect();
    }

    // ── Query ────────────────────────────────────────────────────────────────

    public QueryResult[] query(
            String table, float[] queryVector,
            int topK, float threshold) {
        ensureConnected();
        QueryResult[] results = nativeQuery(handle, table, queryVector, topK, threshold);
        return results != null ? results : new QueryResult[0];
    }

    public QueryResult[] query(String table, float[] queryVector) {
        return query(table, queryVector, 10, 0.8f);
    }

    // ── Write ────────────────────────────────────────────────────────────────

    public void write(String table, float[] vector, String textRepr) {
        ensureConnected();
        nativeWrite(handle, table, vector, textRepr != null ? textRepr : "");
    }

    public void write(String table, float[] vector) {
        write(table, vector, "");
    }

    // ── Status ───────────────────────────────────────────────────────────────

    public boolean isConnected() { return connected; }
    public String  mode()        { return "jni"; }

    // ── Internal ─────────────────────────────────────────────────────────────

    private void ensureConnected() {
        if (!connected) {
            throw new PrismDriverException(
                    "PrismDriver is not connected. Call connect() first.");
        }
    }
}
