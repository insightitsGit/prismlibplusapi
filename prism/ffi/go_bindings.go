// Package prismdriver provides Go cgo bindings for the PrismDriver C ABI.
//
// Build tags: requires the compiled prism_driver shared library.
// Set CGO_LDFLAGS to point to the library:
//
//	CGO_LDFLAGS="-L/path/to/prism/lib -lprism_driver -Wl,-rpath,/path/to/prism/lib"
//	go build ./...
//
// Usage:
//
//	driver, err := prismdriver.Connect("db-proxy-1", 50051, "acme-corp", "")
//	if err != nil { log.Fatal(err) }
//	defer driver.Close()
//
//	results, err := driver.Query("orders", embedding, prismdriver.QueryOpts{TopK: 10})
//	if err != nil { log.Fatal(err) }
//
//	for _, r := range results {
//		fmt.Printf("%.4f  %s\n", r.Score, r.TextRepr)
//	}
//
//	if err := driver.Write("orders", embedding, "order #42"); err != nil {
//		log.Fatal(err)
//	}
package prismdriver

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -lprism_driver

#include "prism_driver.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Driver wraps a prism_driver_t connection handle.
type Driver struct {
	handle *C.prism_driver_t
}

// QueryResult holds one row returned by a vector similarity query.
type QueryResult struct {
	EventID  string
	RowID    string
	Score    float32
	TextRepr string
	Vector   []float32
}

// QueryOpts configures a Query call.
type QueryOpts struct {
	TopK      int
	Threshold float32
}

// Version returns the version string of the loaded prism_driver library.
func Version() string {
	return C.GoString(C.prism_version())
}

// Connect opens a connection to the Server Wrapper.
//
// host:     Wrapper hostname or IP.
// port:     Wrapper gRPC port (typically 50051).
// tenantID: Tenant identifier (empty string for none).
// tlsCert:  Path to PEM certificate for mTLS (empty string for plaintext).
func Connect(host string, port int, tenantID, tlsCert string) (*Driver, error) {
	cHost     := C.CString(host)
	cTenant   := C.CString(tenantID)
	var cTLS  *C.char
	if tlsCert != "" {
		cTLS = C.CString(tlsCert)
		defer C.free(unsafe.Pointer(cTLS))
	}
	defer C.free(unsafe.Pointer(cHost))
	defer C.free(unsafe.Pointer(cTenant))

	handle := C.prism_connect(cHost, C.int(port), cTenant, cTLS)
	if handle == nil {
		return nil, fmt.Errorf("prism_connect failed: %s",
			C.GoString(C.prism_last_error(nil)))
	}
	return &Driver{handle: handle}, nil
}

// Close disconnects and frees all resources.
func (d *Driver) Close() error {
	if d.handle == nil {
		return nil
	}
	rc := C.prism_disconnect(d.handle)
	d.handle = nil
	if rc != 0 {
		return fmt.Errorf("prism_disconnect failed: code %d", rc)
	}
	return nil
}

// Query runs a vector similarity query against the specified table.
func (d *Driver) Query(table string, vector []float32, opts QueryOpts) ([]QueryResult, error) {
	if d.handle == nil {
		return nil, fmt.Errorf("not connected")
	}
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 0.8
	}

	cTable := C.CString(table)
	defer C.free(unsafe.Pointer(cTable))

	cVec := (*C.float)(C.CBytes(float32SliceToBytes(vector)))
	defer C.free(unsafe.Pointer(cVec))

	outBuf  := make([]C.prism_result_t, opts.TopK)
	outCount := C.int(0)

	rc := C.prism_query(
		d.handle,
		cTable,
		cVec,
		C.int(len(vector)),
		C.int(opts.TopK),
		C.float(opts.Threshold),
		(*C.prism_result_t)(unsafe.Pointer(&outBuf[0])),
		&outCount,
	)
	if rc != 0 {
		return nil, fmt.Errorf("prism_query failed: %s",
			C.GoString(C.prism_last_error(d.handle)))
	}

	n       := int(outCount)
	results := make([]QueryResult, n)
	for i := 0; i < n; i++ {
		r := &outBuf[i]
		var vec []float32
		if r.vector_dim > 0 && r.vector != nil {
			vec = make([]float32, r.vector_dim)
			for j := 0; j < int(r.vector_dim); j++ {
				vec[j] = float32(*(*C.float)(unsafe.Pointer(
					uintptr(unsafe.Pointer(r.vector)) + uintptr(j)*4,
				)))
			}
		}
		results[i] = QueryResult{
			EventID:  C.GoString(&r.event_id[0]),
			RowID:    C.GoString(&r.row_id[0]),
			Score:    float32(r.score),
			TextRepr: C.GoString(&r.text_repr[0]),
			Vector:   vec,
		}
		C.prism_free_result_vector(r)
	}
	return results, nil
}

// Write sends a write-behind vector write to the DB via the Server Wrapper.
func (d *Driver) Write(table string, vector []float32, textRepr string) error {
	if d.handle == nil {
		return fmt.Errorf("not connected")
	}

	cTable    := C.CString(table)
	cTextRepr := C.CString(textRepr)
	defer C.free(unsafe.Pointer(cTable))
	defer C.free(unsafe.Pointer(cTextRepr))

	cVec := (*C.float)(C.CBytes(float32SliceToBytes(vector)))
	defer C.free(unsafe.Pointer(cVec))

	rc := C.prism_write(d.handle, cTable, cVec, C.int(len(vector)), cTextRepr)
	if rc != 0 {
		return fmt.Errorf("prism_write failed: %s",
			C.GoString(C.prism_last_error(d.handle)))
	}
	return nil
}

// float32SliceToBytes reinterprets a []float32 as []byte without copying.
func float32SliceToBytes(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	b := make([]byte, len(f)*4)
	for i, v := range f {
		bits := *(*uint32)(unsafe.Pointer(&v))
		b[i*4+0] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}
