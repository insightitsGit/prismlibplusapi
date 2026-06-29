/**
 * prism_driver.h — PrismDriver C ABI
 *
 * This header defines the public C interface exported by prism_driver.so /
 * prism_driver.dll.  All language bindings (C#, PHP, Go, Java) target this
 * ABI so that one compiled binary serves every language on a given OS/arch.
 *
 * Thread-safety: all functions are thread-safe.  A single prism_driver_t
 * may be shared across threads; the library uses internal locking.
 *
 * Error handling: most functions return an int.  0 means success.
 * Negative values are error codes (see PRISM_E_* constants below).
 * Call prism_last_error() to get a human-readable message after any failure.
 *
 * Compile:
 *   cmake -S prism/ffi -B build/ffi -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build/ffi --config Release
 */

#ifndef PRISM_DRIVER_H
#define PRISM_DRIVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Platform export macro ───────────────────────────────────────────────── */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef PRISM_DRIVER_EXPORTS
    #define PRISM_API __declspec(dllexport)
  #else
    #define PRISM_API __declspec(dllimport)
  #endif
#else
  #define PRISM_API __attribute__((visibility("default")))
#endif

/* ── Version ─────────────────────────────────────────────────────────────── */
#define PRISM_DRIVER_VERSION_MAJOR 0
#define PRISM_DRIVER_VERSION_MINOR 2
#define PRISM_DRIVER_VERSION_PATCH 0

/* ── Error codes ─────────────────────────────────────────────────────────── */
#define PRISM_OK              0
#define PRISM_E_INVALID_ARG  -1
#define PRISM_E_NOT_CONNECTED -2
#define PRISM_E_TRANSPORT    -3
#define PRISM_E_TIMEOUT      -4
#define PRISM_E_AUTH         -5
#define PRISM_E_INTERNAL     -99

/* ── Opaque handle ───────────────────────────────────────────────────────── */
typedef struct prism_driver_t prism_driver_t;

/* ── Query result ────────────────────────────────────────────────────────── */
typedef struct {
    char    event_id[37];   /* UUID string, null-terminated */
    char    row_id[128];    /* row primary key, null-terminated */
    float   score;          /* similarity score [0, 1] */
    char    text_repr[512]; /* human-readable row text, null-terminated */
    float*  vector;         /* float32[dim] or NULL if not requested */
    int     vector_dim;     /* length of vector array, or 0 */
} prism_result_t;

/* ── Connection ──────────────────────────────────────────────────────────── */

/**
 * Open a connection to the Server Wrapper.
 *
 * @param host       Wrapper hostname or IP (null-terminated UTF-8).
 * @param port       Wrapper gRPC port (typically 50051).
 * @param tenant_id  Tenant identifier (null-terminated UTF-8, or "" for none).
 * @param tls_cert   Path to PEM certificate for mTLS, or NULL for plaintext.
 *
 * @return  Non-NULL handle on success, NULL on failure.
 *          Call prism_last_error(NULL) to get the error message.
 */
PRISM_API prism_driver_t* prism_connect(
    const char* host,
    int         port,
    const char* tenant_id,
    const char* tls_cert   /* nullable */
);

/**
 * Close the connection and free all resources.
 * The handle is invalid after this call.
 *
 * @return  PRISM_OK, or a PRISM_E_* error code.
 */
PRISM_API int prism_disconnect(prism_driver_t* handle);

/* ── Query ───────────────────────────────────────────────────────────────── */

/**
 * Run a vector similarity query.
 *
 * @param handle     Connection handle from prism_connect().
 * @param table      Target table name (null-terminated).
 * @param vector     Query vector, float32[dim].
 * @param dim        Vector dimensionality (must match wrapper target_dim).
 * @param top_k      Maximum number of results.
 * @param threshold  Minimum similarity score [0, 1].
 * @param out        Caller-allocated array of at least top_k prism_result_t.
 * @param out_count  Set to the actual number of results written to out[].
 *
 * @return  PRISM_OK on success, PRISM_E_* on failure.
 */
PRISM_API int prism_query(
    prism_driver_t* handle,
    const char*     table,
    const float*    vector,
    int             dim,
    int             top_k,
    float           threshold,
    prism_result_t* out,
    int*            out_count
);

/* ── Write ───────────────────────────────────────────────────────────────── */

/**
 * Send a write-behind vector write to the DB via the Server Wrapper.
 * Returns as soon as the write is queued in the wrapper's write buffer;
 * the actual DB flush is asynchronous.
 *
 * @param handle     Connection handle.
 * @param table      Target table name (null-terminated).
 * @param vector     Row embedding, float32[dim].
 * @param dim        Vector dimensionality.
 * @param text_repr  Human-readable text form of the row (null-terminated).
 *
 * @return  PRISM_OK on success, PRISM_E_* on failure.
 */
PRISM_API int prism_write(
    prism_driver_t* handle,
    const char*     table,
    const float*    vector,
    int             dim,
    const char*     text_repr   /* nullable */
);

/* ── Utilities ───────────────────────────────────────────────────────────── */

/**
 * Return the last error message for this handle as a null-terminated string.
 * Pass NULL for errors that occurred before a handle was created.
 * The returned pointer is owned by the library — do not free it.
 */
PRISM_API const char* prism_last_error(prism_driver_t* handle /* nullable */);

/**
 * Return the library version string (e.g. "0.2.0").
 * The returned pointer is owned by the library.
 */
PRISM_API const char* prism_version(void);

/**
 * Free a vector allocated inside a prism_result_t by the library.
 * Call this on each result in out[] before freeing the result array.
 */
PRISM_API void prism_free_result_vector(prism_result_t* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PRISM_DRIVER_H */
