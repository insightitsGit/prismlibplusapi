<?php
/**
 * PrismDriver PHP FFI Bindings
 *
 * Requires: PHP 8.0+ with FFI extension enabled (extension=ffi in php.ini)
 *
 * Usage:
 *   require_once 'php_bindings.php';
 *
 *   $driver = new PrismDriver(host: 'db-proxy-1', port: 50051, tenantId: 'acme');
 *   $driver->connect();
 *
 *   $results = $driver->query(table: 'orders', vector: $embedding, topK: 10);
 *   foreach ($results as $r) {
 *       echo "{$r['score']}  {$r['text_repr']}\n";
 *   }
 *
 *   $driver->write(table: 'orders', vector: $embedding, textRepr: 'order #42');
 *   $driver->disconnect();
 *
 * Install: copy the shared library for your platform to a directory on
 *   LD_LIBRARY_PATH (Linux), DYLD_LIBRARY_PATH (macOS), or PATH (Windows).
 *   Set PRISM_DRIVER_LIB env to override the library path.
 */

declare(strict_types=1);

class PrismDriverException extends RuntimeException {}

class PrismDriver
{
    private \FFI        $ffi;
    private mixed       $handle = null;
    private bool        $connected = false;

    private const C_HEADER = <<<'C'
        typedef struct prism_driver_t prism_driver_t;

        typedef struct {
            char  event_id[37];
            char  row_id[128];
            float score;
            char  text_repr[512];
            float* vector;
            int   vector_dim;
        } prism_result_t;

        prism_driver_t* prism_connect(
            const char* host, int port,
            const char* tenant_id, const char* tls_cert);

        int prism_disconnect(prism_driver_t* handle);

        int prism_query(
            prism_driver_t* handle,
            const char* table,
            const float* vector,
            int dim, int top_k, float threshold,
            prism_result_t* out, int* out_count);

        int prism_write(
            prism_driver_t* handle,
            const char* table,
            const float* vector,
            int dim, const char* text_repr);

        const char* prism_last_error(prism_driver_t* handle);
        const char* prism_version(void);
        void prism_free_result_vector(prism_result_t* result);
    C;

    public function __construct(
        private readonly string  $host     = 'localhost',
        private readonly int     $port     = 50051,
        private readonly string  $tenantId = '',
        private readonly ?string $tlsCert  = null,
    ) {
        $lib = getenv('PRISM_DRIVER_LIB') ?: $this->detectLib();
        $this->ffi = \FFI::cdef(self::C_HEADER, $lib);
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    public function connect(): void
    {
        $this->handle = $this->ffi->prism_connect(
            $this->host,
            $this->port,
            $this->tenantId,
            $this->tlsCert,
        );

        if ($this->handle === null) {
            throw new PrismDriverException(
                'prism_connect failed: ' . $this->lastError()
            );
        }
        $this->connected = true;
    }

    public function disconnect(): void
    {
        if ($this->connected && $this->handle !== null) {
            $this->ffi->prism_disconnect($this->handle);
            $this->handle    = null;
            $this->connected = false;
        }
    }

    public function __destruct()
    {
        $this->disconnect();
    }

    // ── Query ──────────────────────────────────────────────────────────────

    /**
     * @param float[] $vector  Query embedding (float32 values).
     * @return array<array{event_id:string,row_id:string,score:float,text_repr:string}>
     */
    public function query(
        string $table,
        array  $vector,
        int    $topK      = 10,
        float  $threshold = 0.8,
    ): array {
        $this->ensureConnected();

        $dim  = count($vector);
        $cVec = $this->toFloatArray($vector);

        $out      = $this->ffi->new("prism_result_t[$topK]");
        $outCount = $this->ffi->new('int');

        $rc = $this->ffi->prism_query(
            $this->handle,
            $table,
            $cVec,
            $dim,
            $topK,
            $threshold,
            $out,
            \FFI::addr($outCount),
        );

        if ($rc !== 0) {
            throw new PrismDriverException('prism_query failed: ' . $this->lastError());
        }

        $results = [];
        for ($i = 0; $i < $outCount->cdata; $i++) {
            $r = $out[$i];
            $results[] = [
                'event_id'  => \FFI::string($r->event_id),
                'row_id'    => \FFI::string($r->row_id),
                'score'     => $r->score,
                'text_repr' => \FFI::string($r->text_repr),
            ];
            $this->ffi->prism_free_result_vector(\FFI::addr($r));
        }

        return $results;
    }

    // ── Write ──────────────────────────────────────────────────────────────

    /** @param float[] $vector */
    public function write(string $table, array $vector, string $textRepr = ''): void
    {
        $this->ensureConnected();

        $dim  = count($vector);
        $cVec = $this->toFloatArray($vector);

        $rc = $this->ffi->prism_write($this->handle, $table, $cVec, $dim, $textRepr);
        if ($rc !== 0) {
            throw new PrismDriverException('prism_write failed: ' . $this->lastError());
        }
    }

    // ── Utilities ──────────────────────────────────────────────────────────

    public static function version(): string
    {
        // version() is static so we need a temporary FFI instance
        $lib = getenv('PRISM_DRIVER_LIB') ?: '';
        $ffi = \FFI::cdef('const char* prism_version(void);', $lib);
        return \FFI::string($ffi->prism_version());
    }

    public function isConnected(): bool
    {
        return $this->connected;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    private function ensureConnected(): void
    {
        if (!$this->connected) {
            throw new PrismDriverException('Not connected. Call connect() first.');
        }
    }

    private function lastError(): string
    {
        $ptr = $this->ffi->prism_last_error($this->handle);
        return $ptr !== null ? \FFI::string($ptr) : 'unknown error';
    }

    /** @param float[] $values */
    private function toFloatArray(array $values): mixed
    {
        $n   = count($values);
        $arr = $this->ffi->new("float[$n]");
        foreach ($values as $i => $v) {
            $arr[$i] = (float) $v;
        }
        return $arr;
    }

    private function detectLib(): string
    {
        return match (PHP_OS_FAMILY) {
            'Windows' => 'prism_driver.dll',
            'Darwin'  => 'libprism_driver.dylib',
            default   => 'libprism_driver.so',
        };
    }
}
