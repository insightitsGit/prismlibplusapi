// PrismDriver.cs — C# P/Invoke bindings for the Prism DLL Driver
//
// NuGet package: PrismLib.Driver (ships prism_driver.dll as a native asset)
//
// Usage:
//   using Prism.Driver;
//
//   var driver = new PrismDriver("db-proxy-1", 50051, tenantId: "acme");
//   await driver.ConnectAsync();
//
//   var results = await driver.QueryAsync("orders", myEmbedding, topK: 10);
//   foreach (var r in results)
//       Console.WriteLine($"  {r.Score:F4}  {r.TextRepr}");
//
//   await driver.WriteAsync("orders", newEmbedding, "order #42 for user jane");
//   await driver.DisconnectAsync();
//
// .csproj — add native assets:
//   <ItemGroup>
//     <None Include="native\win-x64\prism_driver.dll"
//           Pack="true" PackagePath="runtimes\win-x64\native" />
//     <None Include="native\linux-x64\libprism_driver.so"
//           Pack="true" PackagePath="runtimes\linux-x64\native" />
//     <None Include="native\osx-arm64\libprism_driver.dylib"
//           Pack="true" PackagePath="runtimes\osx-arm64\native" />
//   </ItemGroup>

using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Prism.Driver
{
    // ── Native result struct ─────────────────────────────────────────────────
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal struct NativeResult
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 37)]
        public string EventId;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string RowId;

        public float Score;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string TextRepr;

        public IntPtr Vector;     // float* — or IntPtr.Zero
        public int    VectorDim;
    }

    // ── Managed result ───────────────────────────────────────────────────────
    public sealed class QueryResult
    {
        public string  EventId   { get; init; } = "";
        public string  RowId     { get; init; } = "";
        public float   Score     { get; init; }
        public string  TextRepr  { get; init; } = "";
        public float[] Vector    { get; init; } = Array.Empty<float>();
    }

    // ── P/Invoke declarations ────────────────────────────────────────────────
    internal static class NativeMethods
    {
        private const string Lib = "prism_driver";

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl,
                   CharSet = CharSet.Ansi)]
        public static extern IntPtr prism_connect(
            string host, int port, string tenantId, string? tlsCert);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int prism_disconnect(IntPtr handle);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl,
                   CharSet = CharSet.Ansi)]
        public static extern int prism_query(
            IntPtr         handle,
            string         table,
            float[]        vector,
            int            dim,
            int            topK,
            float          threshold,
            [Out] NativeResult[] outResults,
            out int        outCount);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl,
                   CharSet = CharSet.Ansi)]
        public static extern int prism_write(
            IntPtr handle, string table,
            float[] vector, int dim, string? textRepr);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr prism_last_error(IntPtr handle);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr prism_version();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void prism_free_result_vector(ref NativeResult result);
    }

    // ── Public driver ────────────────────────────────────────────────────────
    public sealed class PrismDriver : IAsyncDisposable
    {
        private readonly string  _host;
        private readonly int     _port;
        private readonly string  _tenantId;
        private readonly string? _tlsCert;
        private IntPtr           _handle = IntPtr.Zero;

        public bool IsConnected => _handle != IntPtr.Zero;
        public string Mode      => "dll";

        public PrismDriver(string host, int port = 50051,
                           string tenantId = "", string? tlsCert = null)
        {
            _host     = host;
            _port     = port;
            _tenantId = tenantId;
            _tlsCert  = tlsCert;
        }

        // ── Lifecycle ────────────────────────────────────────────────────────
        public Task ConnectAsync()
        {
            _handle = NativeMethods.prism_connect(_host, _port, _tenantId, _tlsCert);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException(
                    $"prism_connect failed: {LastError()}");
            return Task.CompletedTask;
        }

        public Task DisconnectAsync()
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.prism_disconnect(_handle);
                _handle = IntPtr.Zero;
            }
            return Task.CompletedTask;
        }

        public async ValueTask DisposeAsync() => await DisconnectAsync();

        // ── Query ────────────────────────────────────────────────────────────
        public Task<QueryResult[]> QueryAsync(
            string  table,
            float[] queryVector,
            int     topK      = 10,
            float   threshold = 0.8f)
        {
            EnsureConnected();
            var outBuf = new NativeResult[topK];
            int rc = NativeMethods.prism_query(
                _handle, table, queryVector, queryVector.Length,
                topK, threshold, outBuf, out int count);

            if (rc != 0)
                throw new InvalidOperationException($"prism_query failed: {LastError()}");

            var results = new QueryResult[count];
            for (int i = 0; i < count; i++)
            {
                float[] vec = Array.Empty<float>();
                if (outBuf[i].Vector != IntPtr.Zero && outBuf[i].VectorDim > 0)
                {
                    vec = new float[outBuf[i].VectorDim];
                    Marshal.Copy(outBuf[i].Vector, vec, 0, outBuf[i].VectorDim);
                }
                results[i] = new QueryResult
                {
                    EventId  = outBuf[i].EventId,
                    RowId    = outBuf[i].RowId,
                    Score    = outBuf[i].Score,
                    TextRepr = outBuf[i].TextRepr,
                    Vector   = vec,
                };
                NativeMethods.prism_free_result_vector(ref outBuf[i]);
            }
            return Task.FromResult(results);
        }

        // ── Write ────────────────────────────────────────────────────────────
        public Task WriteAsync(string table, float[] vector, string textRepr = "")
        {
            EnsureConnected();
            int rc = NativeMethods.prism_write(
                _handle, table, vector, vector.Length, textRepr);
            if (rc != 0)
                throw new InvalidOperationException($"prism_write failed: {LastError()}");
            return Task.CompletedTask;
        }

        // ── Utilities ────────────────────────────────────────────────────────
        public static string Version()
            => Marshal.PtrToStringAnsi(NativeMethods.prism_version()) ?? "";

        private string LastError()
            => Marshal.PtrToStringAnsi(NativeMethods.prism_last_error(_handle)) ?? "";

        private void EnsureConnected()
        {
            if (!IsConnected)
                throw new InvalidOperationException(
                    "PrismDriver is not connected. Call ConnectAsync() first.");
        }
    }
}
