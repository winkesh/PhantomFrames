// Batch JPEG codec for Phantom Frames tiles (libjpeg-turbo / TurboJPEG 3).
//
// A frame is hundreds of independent 40x40 JPEG tiles (576 at 720p, 1296 at 1080p). Coding them one Python call
// at a time costs more in per-call overhead than in JPEG work, and holds the GIL throughout. These functions code
// a whole frame's tiles in one call, on several native threads, with the GIL released.
//
//   encode_tiles(frame, tile_ids, tile, quality, threads) -> list[bytes]
//       frame: C-contiguous HxWx3 uint8 BGR (e.g. a numpy array). 4:2:0, baseline, like cv2.imencode.
//   decode_tiles(jpegs, out, tile, threads) -> bytes
//       jpegs: sequence of bytes-like; out: writable C-contiguous buffer of >= len(jpegs)*tile*tile*3 bytes,
//       filled with BGR tiles in order. Returns one byte per tile: 1 decoded, 0 corrupt / wrong size (left as is).
//
// Build: python -m tools.build_native
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <turbojpeg.h>

namespace {

struct Out {
    unsigned char *buf = nullptr;
    size_t size = 0;
    std::string err;
};

template <class Fn>
void run_parallel(int n, int threads, Fn fn) {
    // work is handed out in small chunks so faster threads take more of it
    threads = std::max(1, std::min(threads, (n + 15) / 16));
    std::atomic<int> next(0);
    auto worker = [&](int) {
        fn([&](int &lo, int &hi) {
            lo = next.fetch_add(16);
            hi = std::min(n, lo + 16);
            return lo < n;
        });
    };
    if (threads == 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (int t = 1; t < threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto &th : pool) th.join();
}

PyObject *encode_tiles(PyObject *, PyObject *args) {
    PyObject *frame_obj, *ids_obj;
    int tile, quality, threads;
    if (!PyArg_ParseTuple(args, "OOiii", &frame_obj, &ids_obj, &tile, &quality, &threads)) return nullptr;
    Py_buffer fb;
    if (PyObject_GetBuffer(frame_obj, &fb, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) return nullptr;
    if (fb.ndim != 3 || fb.shape[2] != 3 || fb.itemsize != 1) {
        PyBuffer_Release(&fb);
        PyErr_SetString(PyExc_ValueError, "frame must be a C-contiguous HxWx3 uint8 array");
        return nullptr;
    }
    const int H = (int)fb.shape[0], W = (int)fb.shape[1], tiles_x = W / tile, tiles_y = H / tile;
    PyObject *seq = PySequence_Fast(ids_obj, "tile_ids must be a sequence");
    if (!seq) {
        PyBuffer_Release(&fb);
        return nullptr;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    std::vector<int> ids(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (v < 0 || v >= (long)tiles_x * tiles_y) {
            Py_DECREF(seq);
            PyBuffer_Release(&fb);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_IndexError, "tile id out of range");
            return nullptr;
        }
        ids[i] = (int)v;
    }
    Py_DECREF(seq);
    std::vector<Out> outs(n);
    const unsigned char *base = static_cast<const unsigned char *>(fb.buf);
    const int pitch = W * 3;

    Py_BEGIN_ALLOW_THREADS
    run_parallel((int)n, threads, [&](auto next_chunk) {
        tjhandle h = tj3Init(TJINIT_COMPRESS);
        if (!h) return;
        tj3Set(h, TJPARAM_QUALITY, quality);
        tj3Set(h, TJPARAM_SUBSAMP, TJSAMP_420);
        int lo, hi;
        while (next_chunk(lo, hi)) {
            for (int i = lo; i < hi; ++i) {
                const int ty = ids[i] / tiles_x, tx = ids[i] % tiles_x;
                const unsigned char *src = base + (size_t)ty * tile * pitch + (size_t)tx * tile * 3;
                if (tj3Compress8(h, src, tile, pitch, tile, TJPF_BGR, &outs[i].buf, &outs[i].size) != 0)
                    outs[i].err = tj3GetErrorStr(h);
            }
        }
        tj3Destroy(h);
    });
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&fb);
    PyObject *result = PyList_New(n);
    std::string err;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!outs[i].buf && err.empty()) err = outs[i].err.empty() ? "encoder could not start" : outs[i].err;
        PyObject *b = outs[i].buf ? PyBytes_FromStringAndSize((const char *)outs[i].buf, (Py_ssize_t)outs[i].size)
                                  : (Py_INCREF(Py_None), Py_None);
        if (outs[i].buf) tj3Free(outs[i].buf);
        if (result) PyList_SET_ITEM(result, i, b);
        else Py_XDECREF(b);
    }
    if (!err.empty()) {
        Py_XDECREF(result);
        PyErr_SetString(PyExc_RuntimeError, ("JPEG encode failed: " + err).c_str());
        return nullptr;
    }
    return result;
}

PyObject *decode_tiles(PyObject *, PyObject *args) {
    PyObject *jpegs_obj, *out_obj;
    int tile, threads;
    if (!PyArg_ParseTuple(args, "OOii", &jpegs_obj, &out_obj, &tile, &threads)) return nullptr;
    PyObject *seq = PySequence_Fast(jpegs_obj, "jpegs must be a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    const size_t tile_bytes = (size_t)tile * tile * 3;
    Py_buffer ob;
    if (PyObject_GetBuffer(out_obj, &ob, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) {
        Py_DECREF(seq);
        return nullptr;
    }
    if ((size_t)ob.len < (size_t)n * tile_bytes) {
        PyBuffer_Release(&ob);
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "out buffer too small for len(jpegs) tiles");
        return nullptr;
    }
    std::vector<Py_buffer> in(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (PyObject_GetBuffer(PySequence_Fast_GET_ITEM(seq, i), &in[i], PyBUF_SIMPLE) < 0) {
            for (Py_ssize_t j = 0; j < i; ++j) PyBuffer_Release(&in[j]);
            PyBuffer_Release(&ob);
            Py_DECREF(seq);
            return nullptr;
        }
    }
    std::vector<unsigned char> ok(n, 0);
    unsigned char *dst = static_cast<unsigned char *>(ob.buf);

    Py_BEGIN_ALLOW_THREADS
    run_parallel((int)n, threads, [&](auto next_chunk) {
        tjhandle h = tj3Init(TJINIT_DECOMPRESS);
        if (!h) return;
        int lo, hi;
        while (next_chunk(lo, hi)) {
            for (int i = lo; i < hi; ++i) {
                const unsigned char *src = static_cast<const unsigned char *>(in[i].buf);
                const size_t len = (size_t)in[i].len;
                if (tj3DecompressHeader(h, src, len) != 0) continue;
                if (tj3Get(h, TJPARAM_JPEGWIDTH) != tile || tj3Get(h, TJPARAM_JPEGHEIGHT) != tile) continue;
                if (tj3Decompress8(h, src, len, dst + (size_t)i * tile_bytes, tile * 3, TJPF_BGR) == 0) ok[i] = 1;
            }
        }
        tj3Destroy(h);
    });
    Py_END_ALLOW_THREADS

    for (auto &b : in) PyBuffer_Release(&b);
    PyBuffer_Release(&ob);
    Py_DECREF(seq);
    return PyBytes_FromStringAndSize((const char *)ok.data(), n);
}

PyMethodDef methods[] = {
    {"encode_tiles", encode_tiles, METH_VARARGS, "encode_tiles(frame, tile_ids, tile, quality, threads) -> list[bytes]"},
    {"decode_tiles", decode_tiles, METH_VARARGS, "decode_tiles(jpegs, out, tile, threads) -> bytes of ok flags"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_tilecodec", "Batch JPEG tile codec (libjpeg-turbo)", -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__tilecodec(void) { return PyModule_Create(&module); }
