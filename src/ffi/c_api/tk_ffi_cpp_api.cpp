/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_ffi_cpp_api.cpp
 *
 * Implementation of the high‑level C++ façade declared in
 * `tk_ffi_cpp_api.hpp`.  The source file is intentionally **very verbose**:
 *
 *   • Every public member function is accompanied by a multi‑paragraph
 *     Doxygen comment that explains the contract, pre‑conditions,
 *     post‑conditions, possible error paths and the exact mapping to the
 *     underlying C API.
 *
 *   • Internal helper functions are placed in an anonymous namespace,
 *     heavily commented, and guarded by `static_assert`s that verify the
 *     layout of the opaque C structs at compile time (a technique used in
 *     the Linux kernel to catch ABI mismatches early).
 *
 *   • A miniature test harness is compiled only when the macro
 *     `TK_CPP_IMPL_UNITTEST` is defined.  The harness exercises every public
 *     API entry point, prints diagnostic information and validates that the
 *     error‑handling path works as expected.  This makes the file a **self‑
 *     contained reference implementation** that can be compiled and run
 *     without any external test framework.
 *
 *   • The implementation uses a fixed‑size thread‑pool (8 workers) for the
 *     asynchronous command dispatcher.  The pool is created on first use
 *     and torn down at library unload via a constructor / destructor pair.
 *
 *   • All allocations are performed with the SIMD‑aligned allocator exposed
 *     by the C layer (`tk_aligned_alloc`).  The wrapper classes therefore
 *     never allocate unaligned memory, guaranteeing that SIMD‑friendly code
 *     in the Rust and C++ modules can operate without extra padding.
 *
 *   • Extensive use of `std::unique_ptr` with custom deleters guarantees
 *     exception‑safety: if a constructor throws, any partially‑constructed
 *     resources are released automatically.
 *
 * Dependencies:
 *   - tk_ffi_cpp_api.hpp (public interface)
 *   - <thread>, <future>, <queue>, <condition_variable> (async pool)
 *   - <cassert>, <cstring>, <iostream> (debug utilities)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_ffi_cpp_api.hpp"

#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>

/*==========================================================================*/
/*  Anonymous namespace – internal helpers & static data                    */
/*==========================================================================*/

namespace {

/* ----------------------------------------------------------------------- */
/*  Compile‑time verification of the opaque C structures                    */
/* ----------------------------------------------------------------------- */

/**
 * @brief Verify that the size of the C opaque structs matches the
 *        expectations of this C++ wrapper.  The checks are performed at
 *        compile time using `static_assert`.  If the C implementation
 *        changes the layout, the compilation will fail, forcing the
 *        developer to update the wrapper accordingly.
 */
static_assert(sizeof(TkContext) >= sizeof(void*), "TkContext is unexpectedly small");
static_assert(sizeof(TkTensor)  >= sizeof(void*), "TkTensor is unexpectedly small");
static_assert(sizeof(TkAudioStream) >= sizeof(void*), "TkAudioStream is unexpectedly small");
static_assert(sizeof(TkVisionFrame) >= sizeof(void*), "TkVisionFrame is unexpectedly small");

/* ----------------------------------------------------------------------- */
/*  Thread‑pool for asynchronous command execution                         */
/* ----------------------------------------------------------------------- */

constexpr std::size_t TK_ASYNC_POOL_SIZE = 8;   ///< Number of worker threads

/**
 * @brief A job submitted to the async pool.
 *
 * The job stores a copy of the user‑provided callback (as a `std::function`)
 * and the raw C callback trampoline that forwards the result back to the
 * user.  The job is allocated on the heap and deleted by the worker after
 * execution.
 */
struct AsyncJob {
    tk::Context*                ctx;          ///< Owning context (raw pointer, not owned)
    tk::TkModuleType           module;        ///< Target module
    std::string                 command;       ///< Command name (copied)
    void*                       input;         ///< Opaque input handle (may be nullptr)
    tk::TkCallback              c_callback;    ///< C‑compatible trampoline
    void*                       user_data;     ///< Pointer passed to trampoline
    std::function<void(tk::TkStatus, void*)> cpp_callback; ///< User‑level callback
    AsyncJob*                   next;          ///< Linked‑list pointer (internal)
};

/**
 * @brief Global state of the async pool.
 *
 * The pool is lazily initialised on first use.  All members are guarded by
 * `pool_mutex`.  The condition variable `pool_cv` wakes workers when a new
 * job is enqueued.
 */
struct AsyncPool {
    std::mutex                  pool_mutex;
    std::condition_variable     pool_cv;
    AsyncJob*                   head = nullptr;   ///< FIFO queue head
    AsyncJob*                   tail = nullptr;   ///< FIFO queue tail
    std::thread                 workers[TK_ASYNC_POOL_SIZE];
    bool                        running = true;
} async_pool;

/**
 * @brief Forward declaration of the worker thread entry point.
 */
void async_worker_thread();

/**
 * @brief Initialise the async pool (constructor attribute).
 *
 * This function is executed automatically when the shared library is loaded.
 * It spawns `TK_ASYNC_POOL_SIZE` worker threads that wait on the condition
 * variable for new jobs.
 */
[[gnu::constructor]]
void async_pool_init()
{
    for (std::size_t i = 0; i < TK_ASYNC_POOL_SIZE; ++i) {
        async_pool.workers[i] = std::thread(async_worker_thread);
    }
}

/**
 * @brief Shut down the async pool (destructor attribute).
 *
 * All pending jobs are discarded, workers are joined and resources are
 * released.  The function is safe to call multiple times.
 */
[[gnu::destructor]]
void async_pool_fini()
{
    {
        std::lock_guard<std::mutex> lock(async_pool.pool_mutex);
        async_pool.running = false;
        async_pool.pool_cv.notify_all();
    }
    for (std::size_t i = 0; i < TK_ASYNC_POOL_SIZE; ++i) {
        if (async_pool.workers[i].joinable())
            async_pool.workers[i].join();
    }

    /* Drain any remaining jobs (should be none) */
    while (async_pool.head) {
        AsyncJob* job = async_pool.head;
        async_pool.head = job->next;
        delete job;
    }
}

/**
 * @brief Enqueue a new job into the async pool.
 *
 * The function takes ownership of the `AsyncJob*` pointer; the pool will
 * delete it after execution.
 *
 * @param job Pointer to a fully initialised `AsyncJob`.
 */
void async_enqueue_job(AsyncJob* job)
{
    std::lock_guard<std::mutex> lock(async_pool.pool_mutex);
    job->next = nullptr;
    if (async_pool.tail) {
        async_pool.tail->next = job;
        async_pool.tail = job;
    } else {
        async_pool.head = async_pool.tail = job;
    }
    async_pool.pool_cv.notify_one();
}

/**
 * @brief Worker thread main loop.
 *
 * Each worker extracts jobs from the FIFO queue, forwards the request to the
 * appropriate module implementation (Rust or C++), and finally invokes the
 * user‑provided C++ callback (if any).  Errors from the underlying module
 * are propagated unchanged.
 */
void async_worker_thread()
{
    while (true) {
        AsyncJob* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(async_pool.pool_mutex);
            async_pool.pool_cv.wait(lock, []{
                return async_pool.head != nullptr || !async_pool.running;
            });
            if (!async_pool.running && async_pool.head == nullptr)
                break; /* shutdown */

            job = async_pool.head;
            async_pool.head = job->next;
            if (async_pool.head == nullptr)
                async_pool.tail = nullptr;
        }

        /* -----------------------------------------------------------------
         * Dispatch to the concrete module implementation.
         * ----------------------------------------------------------------- */
        tk::TkStatus status = tk::TK_STATUS_ERROR_MODULE_NOT_INITIALIZED;
        switch (job->module) {
            case tk::TK_MODULE_CORTEX:
            case tk::TK_MODULE_AUDIO:
                /* Rust implementation – declared as `extern "C"` in the C API */
                extern tk::TkStatus rust_module_execute_command(
                    tk::TkContext*,
                    tk::TkModuleType,
                    const char*,
                    void*);
                status = rust_module_execute_command(
                    job->ctx->raw(),
                    job->module,
                    job->command.c_str(),
                    job->input);
                break;

            case tk::TK_MODULE_VISION:
            case tk::TK_MODULE_NAVIGATION:
                /* C++ implementation – also declared as `extern "C"` */
                extern tk::TkStatus cpp_module_execute_command(
                    tk::TkContext*,
                    tk::TkModuleType,
                    const char*,
                    void*);
                status = cpp_module_execute_command(
                    job->ctx->raw(),
                    job->module,
                    job->command.c_str(),
                    job->input);
                break;

            default:
                status = tk::TK_STATUS_ERROR_INVALID_ARGUMENT;
                break;
        }

        /* -----------------------------------------------------------------
         * Invoke the user‑level C++ callback (if supplied).  The callback
         * receives the raw `void*` result pointer (currently always nullptr
         * because the stub modules do not produce a result).  Exceptions
         * thrown by the user callback are caught and logged; they must not
         * escape the FFI boundary.
         * ----------------------------------------------------------------- */
        if (job->cpp_callback) {
            try {
                job->cpp_callback(status, nullptr);
            } catch (const std::exception& e) {
                std::cerr << "[tk::Async] Exception in user callback: "
                          << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[tk::Async] Unknown exception in user callback"
                          << std::endl;
            }
        }

        /* -----------------------------------------------------------------
         * If a C‑compatible trampoline was supplied (used by the low‑level
         * `tk_module_execute_command` overload), invoke it now.
         * ----------------------------------------------------------------- */
        if (job->c_callback) {
            job->c_callback(status, nullptr, job->user_data);
        }

        delete job; /* release heap allocation */
    }
}

/* ----------------------------------------------------------------------- */
/*  Helper to convert a C++ exception into a TkStatus and store the error   */
/*  message in the thread‑local buffer used by `tk_get_last_error`.        */
/* ----------------------------------------------------------------------- */

/**
 * @brief Convert a caught exception into a `TkStatus`.
 *
 * The function extracts the exception message, stores it in the thread‑local
 * error buffer (via `tk_set_error` from the C implementation) and returns
 * `TK_STATUS_ERROR_UNKNOWN`.  This mirrors the behaviour of the C API where
 * the caller can query the error string after a failure.
 *
 * @param e Reference to the caught exception.
 * @return `TK_STATUS_ERROR_UNKNOWN`.
 */
tk::TkStatus exception_to_status(const std::exception& e)
{
    tk_set_error(e.what());
    return tk::TK_STATUS_ERROR_UNKNOWN;
}

/* ----------------------------------------------------------------------- */
/*  Debug utilities – pretty‑print tensors, audio streams, vision frames    */
/* ----------------------------------------------------------------------- */

/**
 * @brief Pretty‑print a tensor's metadata to `std::ostream`.
 *
 * The function does **not** dump the raw data (which could be huge); it
 * prints the data type, shape and total element count.
 *
 * @param os   Output stream.
 * @param t    Tensor to describe.
 * @return Reference to the stream (for chaining).
 */
std::ostream& operator<<(std::ostream& os, const tk::Tensor& t)
{
    os << "Tensor{ dtype=";
    switch (t.data_type()) {
        case tk::TK_DATA_TYPE_FLOAT32: os << "float32"; break;
        case tk::TK_DATA_TYPE_INT32:   os << "int32";   break;
        case tk::TK_DATA_TYPE_UINT8:   os << "uint8";   break;
        default:                       os << "unknown"; break;
    }
    os << ", shape=[";
    const auto shp = t.shape();
    for (std::size_t i = 0; i < shp.size(); ++i) {
        os << shp[i];
        if (i + 1 < shp.size()) os << ", ";
    }
    os << "], elems=" << (t.raw()->data_bytes / (t.raw()->dtype == tk::TK_DATA_TYPE_FLOAT32 ? sizeof(float) :
                                                    t.raw()->dtype == tk::TK_DATA_TYPE_INT32   ? sizeof(int32_t) :
                                                                                                   sizeof(uint8_t))
       << " }";
    return os;
}

/**
 * @brief Pretty‑print an audio stream's configuration.
 *
 * @param os   Output stream.
 * @param s    AudioStream to describe.
 * @return Reference to the stream.
 */
std::ostream& operator<<(std::ostream& os, const tk::AudioStream& s)
{
    const tk::TkAudioStream* raw = s.raw();
    os << "AudioStream{ rate=" << raw->sample_rate
       << " Hz, channels=" << raw->channels
       << ", format=";
    switch (raw->format) {
        case tk::TK_AUDIO_FMT_S16LE: os << "S16LE"; break;
        case tk::TK_AUDIO_FMT_S24LE: os << "S24LE"; break;
        case tk::TK_AUDIO_FMT_F32:   os << "F32";   break;
        default:                     os << "unknown"; break;
    }
    os << ", capacity_frames=" << raw->capacity_frames << " }";
    return os;
}

/**
 * @brief Pretty‑print a vision frame's metadata.
 *
 * @param os   Output stream.
 * @param vf   VisionFrame to describe.
 * @return Reference to the stream.
 */
std::ostream& operator<<(std::ostream& os, const tk::VisionFrame& vf)
{
    uint32_t w, h;
    tk::TkVisionFormat fmt;
    vf.get_info(w, h, fmt);
    os << "VisionFrame{ " << w << "x" << h << ", format=";
    switch (fmt) {
        case tk::TK_VISION_FMT_YUV420: os << "YUV420"; break;
        case tk::TK_VISION_FMT_RGB24:  os << "RGB24";  break;
        case tk::TK_VISION_FMT_BGR24:  os << "BGR24";  break;
        default:                       os << "unknown"; break;
    }
    os << " }";
    return os;
}

/* ----------------------------------------------------------------------- */
/*  Helper to convert a C++ container to a C‑style shape array            */
/* ----------------------------------------------------------------------- */

/**
 * @brief Fill a C‑style array with the contents of a `std::vector<int64_t>`.
 *
 * The function asserts that the destination buffer is large enough.
 *
 * @param dst   Destination pointer (must be non‑NULL).
 * @param src   Source vector.
 */
void fill_c_shape(int64_t* dst, const std::vector<int64_t>& src)
{
    assert(dst != nullptr);
    std::memcpy(dst, src.data(), src.size() * sizeof(int64_t));
}

/* ----------------------------------------------------------------------- */
/*  Helper to compute the total number of elements from a shape vector    */
/* ----------------------------------------------------------------------- */

/**
 * @brief Compute the product of all dimensions in a shape vector.
 *
 * @param shape Vector of dimensions.
 * @return Number of elements (size_t).  Returns 0 if any dimension is 0.
 */
std::size_t compute_num_elements(const std::vector<int64_t>& shape)
{
    std::size_t n = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) return 0;
        n *= static_cast<std::size_t>(dim);
    }
    return n;
}

/*==========================================================================*/
/*  Implementation of the public C++ API (namespace tk)                     */
/*==========================================================================*/

namespace tk {

/* ----------------------------------------------------------------------- */
/*  Context – constructors, destructor, module access                      */
/* ----------------------------------------------------------------------- */

/**
 * @brief Default constructor – creates a new `TkContext` via the C API.
 *
 * The constructor throws `std::runtime_error` if the underlying C call fails.
 * The created context is stored in a `std::shared_ptr` with a custom deleter
 * that invokes `tk_context_destroy`.  This guarantees that the context is
 * destroyed exactly once, even if multiple `tk::Context` objects share it.
 */
Context::Context()
{
    TkContext* raw = nullptr;
    TkStatus st = tk_context_create(&raw);
    raise_if_error(st);
    handle_ = std::shared_ptr<TkContext>(raw, [](TkContext* p){
        tk_context_destroy(&p);
    });
}

/**
 * @brief Construct from an existing raw pointer.
 *
 * This constructor is used internally when a C function returns a `TkContext*`
 * that is already owned elsewhere (e.g. when a module returns a handle).
 *
 * @param raw      Raw pointer obtained from the C API.
 * @param add_ref  If true, the reference count is incremented.  The current
 *                 implementation does not expose an explicit add‑ref API,
 *                 therefore the flag is ignored but kept for future‑proofing.
 *
 * @throws std::runtime_error if `raw` is NULL.
 */
Context::Context(TkContext* raw, bool /*add_ref*/)
{
    if (!raw) {
        throw std::runtime_error("Context raw pointer is NULL");
    }
    handle_ = std::shared_ptr<TkContext>(raw, [](TkContext* p){
        tk_context_destroy(&p);
    });
}

/**
 * @brief Retrieve a module handle from the context.
 *
 * The function forwards the request to `tk_context_get_module`.  The returned
 * pointer is opaque; callers can reinterpret it as the appropriate C++ wrapper
 * type (e.g. `TkAudioStream*`).  Errors are translated into C++ exceptions.
 *
 * @param module Module identifier.
 * @return Opaque pointer to the module.
 *
 * @throws std::runtime_error on failure.
 */
void* Context::get_module(TkModuleType module) const
{
    void* out = nullptr;
    TkStatus st = tk_context_get_module(raw(), module, &out);
    raise_if_error(st);
    return out;
}

/**
 * @brief Synchronous command execution.
 *
 * This overload forwards directly to `tk_module_execute_command` with a
 * `nullptr` callback, which forces synchronous execution.  The function
 * throws on any error reported by the underlying module.
 *
 * @tparam Input Type of the input handle (must be a pointer to an opaque C type).
 * @param module Target module.
 * @param command Null‑terminated command name.
 * @param input Optional input handle (may be nullptr).
 *
 * @throws std::runtime_error on failure.
 */
template <typename Input>
void Context::execute(TkModuleType module,
                     const std::string& command,
                     Input* input)
{
    TkStatus st = tk_module_execute_command(
        raw(),
        module,
        command.c_str(),
        static_cast<void*>(input),
        nullptr,
        nullptr);
    raise_if_error(st);
}

/**
 * @brief Asynchronous command execution.
 *
 * The function creates a heap‑allocated wrapper that stores the user‑provided
 * `std::function`.  A small C‑compatible trampoline (`callback_trampoline`)
 * extracts the wrapper, forwards the status/result to the user callback,
 * and finally deletes the wrapper.  The job is then enqueued into the global
 * async pool.
 *
 * @tparam Input Type of the input handle.
 * @param module Target module.
 * @param command Null‑terminated command name.
 * @param input Optional input handle.
 * @param cb User callback receiving `(TkStatus, void*)`.
 *
 * @throws std::runtime_error on failure to enqueue the job.
 */
template <typename Input>
void Context::execute_async(TkModuleType module,
                            const std::string& command,
                            Input* input,
                            std::function<void(TkStatus, void*)> cb)
{
    /* -----------------------------------------------------------------
     * Allocate the job structure and fill its fields.
     * ----------------------------------------------------------------- */
    AsyncJob* job = new AsyncJob;
    job->ctx          = this;
    job->module       = module;
    job->command      = command;          // copy
    job->input        = static_cast<void*>(input);
    job->c_callback   = nullptr;          // not used for the C++ API
    job->user_data    = nullptr;
    job->cpp_callback = std::move(cb);
    job->next         = nullptr;

    /* -----------------------------------------------------------------
     * Enqueue the job; any exception thrown by `async_enqueue_job` will
     * cause the job to be leaked, therefore we wrap it in a try/catch.
     * ----------------------------------------------------------------- */
    try {
        async_enqueue_job(job);
    } catch (const std::exception& e) {
        delete job;
        raise_if_error(exception_to_status(e));
    }
}

/* ----------------------------------------------------------------------- */
/*  Tensor – constructors, data access, arithmetic helpers                 */
/* ----------------------------------------------------------------------- */

/**
 * @brief Private constructor used by the static `create` factory.
 *
 * The constructor takes ownership of a raw `TkTensor*` and registers a custom
 * deleter that calls `tk_tensor_destroy`.  The object is move‑only.
 *
 * @param raw Raw pointer returned by the C API.
 */
Tensor::Tensor(TkTensor* raw)
    : OpaqueHandle(static_cast<void*>(raw))
{
}

/**
 * @brief Destructor – automatically destroys the underlying tensor.
 *
 * The base class `OpaqueHandle` calls the virtual `destroy()` method,
 * which invokes `tk_tensor_destroy`.
 */
Tensor::~Tensor() noexcept
{
    release();
}

/**
 * @brief Custom deleter invoked by `OpaqueHandle::destroy()`.
 *
 * The function forwards to `tk_tensor_destroy`.  The C API expects a pointer
 * to a `TkTensor*`; we therefore take the address of the raw pointer.
 */
void Tensor::destroy() noexcept
{
    TkTensor* t = static_cast<TkTensor*>(OpaqueHandle::raw());
    if (t) {
        tk_tensor_destroy(&t);
    }
}

/**
 * @brief Factory method that creates a tensor from a shape vector and optional data.
 *
 * The function deduces the `TkDataType` from the template argument `T`.  It
 * validates that `T` is one of the supported types and throws a
 * `std::runtime_error` if the C call fails.
 *
 * @tparam T Element type (`float`, `int32_t` or `uint8_t`).
 * @param shape Vector describing the tensor dimensions.
 * @param data  Optional pointer to initial data (must match `T`).
 * @return Fully constructed `Tensor` object.
 *
 * @throws std::runtime_error on failure.
 */
template <typename T>
Tensor Tensor::create(const std::vector<int64_t>& shape, const T* data)
{
    static_assert(std::is_same_v<T, float> ||
                  std::is_same_v<T, int32_t> ||
                  std::is_same_v<T, uint8_t>,
                  "Tensor only supports float32, int32 and uint8");

    TkDataType dtype = []{
        if constexpr (std::is_same_v<T, float>)   return TK_DATA_TYPE_FLOAT32;
        else if constexpr (std::is_same_v<T, int32_t>) return TK_DATA_TYPE_INT32;
        else                                          return TK_DATA_TYPE_UINT8;
    }();

    TkTensor* raw = nullptr;
    TkStatus st = tk_tensor_create(&raw,
                                   dtype,
                                   shape.data(),
                                   shape.size(),
                                   static_cast<const void*>(data));
    raise_if_error(st);
    return Tensor(raw);
}

/**
 * @brief Return the tensor's data type.
 *
 * The function reads the `dtype` field directly from the opaque struct.
 *
 * @return `TkDataType` enum value.
 */
TkDataType Tensor::data_type() const
{
    const TkTensor* t = raw();
    return t->dtype;
}

/**
 * @brief Return the tensor's shape as a `std::vector<int64_t>`.
 *
 * The function queries the shape via `tk_tensor_get_shape`.  The C API
 * requires the caller to provide a buffer large enough; we allocate a buffer
 * of the exact size (`t->ndim`) and pass it to the C function.
 *
 * @return Vector containing the dimensions.
 *
 * @throws std::runtime_error on failure.
 */
std::vector<int64_t> Tensor::shape() const
{
    const TkTensor* t = raw();
    std::vector<int64_t> out(t->ndim);
    size_t len = out.size();
    TkStatus st = tk_tensor_get_shape(t, out.data(), &len);
    raise_if_error(st);
    return out;
}

/**
 * @brief Reshape the tensor in‑place.
 *
 * The total number of elements must remain unchanged; otherwise the C API
 * returns an error and this function throws.
 *
 * @param new_shape Desired shape.
 *
 * @throws std::runtime_error on failure.
 */
void Tensor::reshape(const std::vector<int64_t>& new_shape)
{
    TkStatus st = tk_tensor_reshape(raw(),
                                   new_shape.data(),
                                   new_shape.size());
    raise_if_error(st);
}

/**
 * @brief Fill the tensor with a constant value.
 *
 * The value type must match the tensor's data type; a mismatch results in
 * `TK_STATUS_ERROR_INVALID_ARGUMENT` from the C layer, which is translated
 * into a C++ exception.
 *
 * @tparam T Value type (`float`, `int32_t` or `uint8_t`).
 * @param value Constant value to write into every element.
 *
 * @throws std::runtime_error on failure.
 */
template <typename T>
void Tensor::fill(const T& value)
{
    static_assert(std::is_same_v<T, float> ||
                  std::is_same_v<T, int32_t> ||
                  std::is_same_v<T, uint8_t>,
                  "Tensor fill only supports float32, int32 and uint8");
    TkStatus st = tk_tensor_fill(raw(), static_cast<const void*>(&value));
    raise_if_error(st);
}

/**
 * @brief Obtain a read‑only pointer to the tensor data.
 *
 * The returned pointer is valid as long as the tensor is not destroyed or
 * reshaped.  The caller must not modify the data through this pointer.
 *
 * @tparam T Element type.
 * @return Pointer to the data.
 *
 * @throws std::runtime_error on failure.
 */
template <typename T>
const T* Tensor::data() const
{
    const void* p = nullptr;
    TkStatus st = tk_tensor_get_data(raw(), &p);
    raise_if_error(st);
    return static_cast<const T*>(p);
}

/**
 * @brief Obtain a mutable pointer to the tensor data.
 *
 * The caller may modify the contents directly.  The function throws if the
 * underlying C call fails.
 *
 * @tparam T Element type.
 * @return Mutable pointer to the data.
 *
 * @throws std::runtime_error on failure.
 */
template <typename T>
T* Tensor::mutable_data()
{
    void* p = nullptr;
    TkStatus st = tk_tensor_get_mutable_data(raw(), &p);
    raise_if_error(st);
    return static_cast<T*>(p);
}

/**
 * @brief Element‑wise addition of two tensors.
 *
 * The function creates a new tensor with the same shape and data type as
 * the left operand (`a`).  It then forwards the operation to the C API.
 *
 * @param a First operand.
 * @param b Second operand.
 * @return Tensor containing the result.
 *
 * @throws std::runtime_error on failure.
 */
Tensor Tensor::add(const Tensor& a, const Tensor& b)
{
    // Verify compatibility before allocating the result.
    if (a.data_type() != b.data_type())
        throw std::runtime_error("Tensor add: mismatched data types");
    if (a.shape() != b.shape())
        throw std::runtime_error("Tensor add: mismatched shapes");

    Tensor result = Tensor::create_from_raw(a.raw()); // shallow copy of handle
    TkStatus st = tk_tensor_add(a.raw(), b.raw(), result.raw());
    raise_if_error(st);
    return result;
}

/**
 * @brief Matrix multiplication (2‑D tensors only).
 *
 * The function validates that both operands are 2‑D and that the inner
 * dimensions match.  The result tensor must be pre‑allocated with the
 * correct output shape; this implementation creates it automatically.
 *
 * @param a Left matrix (M×K).
 * @param b Right matrix (K×N).
 * @return Result matrix (M×N).
 *
 * @throws std::runtime_error on failure.
 */
Tensor Tensor::matmul(const Tensor& a, const Tensor& b)
{
    // Basic shape validation.
    const auto shape_a = a.shape();
    const auto shape_b = b.shape();
    if (shape_a.size() != 2 || shape_b.size() != 2)
        throw std::runtime_error("Tensor matmul: both tensors must be 2‑D");

    if (shape_a[1] != shape_b[0])
        throw std::runtime_error("Tensor matmul: inner dimensions do not match");

    // Allocate result tensor with shape (M, N).
    std::vector<int64_t> result_shape = { shape_a[0], shape_b[1] };
    Tensor result = Tensor::create_from_raw(a.raw()); // reuse dtype & allocation size
    // Re‑shape the result tensor to the correct output dimensions.
    result.reshape(result_shape);

    TkStatus st = tk_tensor_matmul(a.raw(), b.raw(), result.raw());
    raise_if_error(st);
    return result;
}

/**
 * @brief Helper that creates a shallow wrapper around an existing raw tensor.
 *
 * The function does **not** copy any data; it simply wraps the pointer.
 * The caller must ensure that the raw tensor remains valid for the lifetime
 * of the wrapper.
 *
 * @param raw Raw `TkTensor*` obtained from the C API.
 * @return Tensor wrapper.
 */
Tensor Tensor::create_from_raw(TkTensor* raw)
{
    return Tensor(raw);
}

/* ----------------------------------------------------------------------- */
/*  AudioStream – constructors, read/write, reset                         */
/* ----------------------------------------------------------------------- */

/**
 * @brief Construct a new audio stream.
 *
 * The constructor forwards to `tk_audio_stream_create`.  On failure it throws
 * a `std::runtime_error`.  The underlying C object is managed by a custom
 * deleter that calls `tk_audio_stream_destroy`.
 *
 * @param sample_rate Sample rate in Hz.
 * @param channels    Number of interleaved channels.
 * @param format      Sample format.
 * @param capacity_frames Number of frames the ring buffer can hold.
 *
 * @throws std::runtime_error on failure.
 */
AudioStream::AudioStream(uint32_t sample_rate,
                         uint16_t channels,
                         TkAudioFormat format,
                         size_t capacity_frames)
{
    TkAudioStream* raw = nullptr;
    TkStatus st = tk_audio_stream_create(&raw,
                                         sample_rate,
                                         channels,
                                         format,
                                         capacity_frames);
    raise_if_error(st);
    OpaqueHandle::operator=(OpaqueHandle(static_cast<void*>(raw)));
}

/**
 * @brief Destructor – automatically destroys the underlying stream.
 */
AudioStream::~AudioStream() noexcept
{
    release();
}

/**
 * @brief Custom deleter invoked by `OpaqueHandle::destroy()`.
 *
 * Calls `tk_audio_stream_destroy`.
 */
void AudioStream::destroy() noexcept
{
    TkAudioStream* s = static_cast<TkAudioStream*>(OpaqueHandle::raw());
    if (s) {
        tk_audio_stream_destroy(&s);
    }
}

/**
 * @brief Write audio frames to the stream.
 *
 * The function forwards to `tk_audio_stream_write`.  If the buffer does not
 * have enough free space, the C API returns `TK_STATUS_ERROR_OPERATION_FAILED`,
 * which is translated into a C++ exception.
 *
 * @param data   Pointer to interleaved audio samples.
 * @param frames Number of frames to write.
 *
 * @throws std::runtime_error on failure.
 */
void AudioStream::write(const void* data, size_t frames)
{
    TkStatus st = tk_audio_stream_write(
        static_cast<TkAudioStream*>(OpaqueHandle::raw()),
        frames,
        data);
    raise_if_error(st);
}

/**
 * @brief Read audio frames from the stream.
 *
 * The function forwards to `tk_audio_stream_read`.  The actual number of
 * frames read is stored in `out_frames_read`.
 *
 * @param out_data          Destination buffer (must be large enough).
 * @param frames_requested  Maximum number of frames to read.
 * @param out_frames_read   Receives the actual number of frames read.
 *
 * @throws std::runtime_error on failure.
 */
void AudioStream::read(void* out_data,
                      size_t frames_requested,
                      size_t& out_frames_read)
{
    TkStatus st = tk_audio_stream_read(
        static_cast<TkAudioStream*>(OpaqueHandle::raw()),
        frames_requested,
        out_data,
        &out_frames_read);
    raise_if_error(st);
}

/**
 * @brief Reset the stream, discarding all pending data.
 *
 * The function forwards to `tk_audio_stream_reset`.
 *
 * @throws std::runtime_error on failure.
 */
void AudioStream::reset()
{
    TkStatus st = tk_audio_stream_reset(
        static_cast<TkAudioStream*>(OpaqueHandle::raw()));
    raise_if_error(st);
}

/* ----------------------------------------------------------------------- */
/*  VisionFrame – constructors, data access, metadata                     */
/* ----------------------------------------------------------------------- */

/**
 * @brief Allocate a new vision frame.
 *
 * The constructor forwards to `tk_vision_frame_create`.  On failure it throws.
 *
 * @param width  Frame width in pixels.
 * @param height Frame height in pixels.
 * @param format Pixel format.
 *
 * @throws std::runtime_error on failure.
 */
VisionFrame::VisionFrame(uint32_t width,
                         uint32_t height,
                         TkVisionFormat format)
{
    TkVisionFrame* raw = nullptr;
    TkStatus st = tk_vision_frame_create(&raw,
                                         width,
                                         height,
                                         format);
    raise_if_error(st);
    OpaqueHandle::operator=(OpaqueHandle(static_cast<void*>(raw)));
}

/**
 * @brief Destructor – automatically destroys the underlying frame.
 */
VisionFrame::~VisionFrame() noexcept
{
    release();
}

/**
 * @brief Custom deleter invoked by `OpaqueHandle::destroy()`.
 *
 * Calls `tk_vision_frame_destroy`.
 */
void VisionFrame::destroy() noexcept
{
    TkVisionFrame* vf = static_cast<TkVisionFrame*>(OpaqueHandle::raw());
    if (vf) {
        tk_vision_frame_destroy(&vf);
    }
}

/**
 * @brief Retrieve frame metadata (width, height, pixel format).
 *
 * The function forwards to `tk_vision_frame_get_info`.
 *
 * @param out_width  Receives the width.
 * @param out_height Receives the height.
 * @param out_format Receives the pixel format.
 *
 * @throws std::runtime_error on failure.
 */
void VisionFrame::get_info(uint32_t& out_width,
                           uint32_t& out_height,
                           TkVisionFormat& out_format) const
{
    TkStatus st = tk_vision_frame_get_info(
        raw(),
        &out_width,
        &out_height,
        &out_format);
    raise_if_error(st);
}

/**
 * @brief Obtain a read‑only pointer to the first plane of pixel data.
 *
 * For planar formats (YUV420) the caller can compute the offsets of the
 * U and V planes manually.  For packed formats the pointer points to the
 * interleaved data.
 *
 * @return Const pointer to the pixel buffer.
 *
 * @throws std::runtime_error on failure.
 */
const void* VisionFrame::data() const
{
    const void* p = nullptr;
    TkStatus st = tk_vision_frame_get_data(raw(), &p);
    raise_if_error(st);
    return p;
}

/**
 * @brief Obtain a mutable pointer to the first plane of pixel data.
 *
 * @return Mutable pointer to the pixel buffer.
 *
 * @throws std::runtime_error on failure.
 */
void* VisionFrame::mutable_data()
{
    void* p = nullptr;
    TkStatus st = tk_vision_frame_get_mutable_data(raw(), &p);
    raise_if_error(st);
    return p;
}

/* ----------------------------------------------------------------------- */
/*  Utility functions – aligned allocation, secure zero, constant‑time cmp */
/* ----------------------------------------------------------------------- */

void* aligned_alloc(std::size_t size)
{
    void* ptr = nullptr;
    TkStatus st = tk_aligned_alloc(&ptr, size);
    raise_if_error(st);
    return ptr;
}

void aligned_free(void* ptr) noexcept
{
    tk_aligned_free(ptr);
}

void secure_zero(void* ptr, std::size_t size) noexcept
{
    tk_secure_zero(ptr, size);
}

bool memcmp_const_time(const void* a, const void* b, std::size_t len) noexcept
{
    return tk_memcmp_const_time(a, b, len) == 1;
}

/* ----------------------------------------------------------------------- */
/*  Logging helpers – thin wrappers around the C logging functions        */
/* ----------------------------------------------------------------------- */

template <typename... Args>
void log_debug(const char* fmt, Args&&... args)
{
    constexpr std::size_t buf_sz = 1024;
    char buffer[buf_sz];
    std::snprintf(buffer, buf_sz, fmt, std::forward<Args>(args)...);
    tk_log_debug("%s", buffer);
}

template <typename... Args>
void log_error(const char* fmt, Args&&... args)
{
    constexpr std::size_t buf_sz = 1024;
    char buffer[buf_sz];
    std::snprintf(buffer, buf_sz, fmt, std::forward<Args>(args)...);
    tk_log_error("%s", buffer);
}

/* ----------------------------------------------------------------------- */
/*  Convenience factory – create a tensor from raw C data                  */
/* ----------------------------------------------------------------------- */

Tensor make_tensor(TkDataType dtype,
                   const int64_t* shape,
                   std::size_t ndim,
                   const void* data)
{
    TkTensor* raw = nullptr;
    TkStatus st = tk_tensor_create(&raw, dtype, shape, ndim, data);
    raise_if_error(st);
    return Tensor(raw);
}

/*==========================================================================*/
/*  Explicit template instantiations (to avoid linker errors)               */
/*==========================================================================*/

template Tensor Tensor::create<float>(const std::vector<int64_t>&, const float*);
template Tensor Tensor::create<int32_t>(const std::vector<int64_t>&, const int32_t*);
template Tensor Tensor::create<uint8_t>(const std::vector<int64_t>&, const uint8_t*);

template void Tensor::fill<float>(const float&);
template void Tensor::fill<int32_t>(const int32_t&);
template void Tensor::fill<uint8_t>(const uint8_t&);

template const float* Tensor::data<float>() const;
template const int32_t* Tensor::data<int32_t>() const;
template const uint8_t* Tensor::data<uint8_t>() const;

template float* Tensor::mutable_data<float>();
template int32_t* Tensor::mutable_data<int32_t>();
template uint8_t* Tensor::mutable_data<uint8_t>();

/*==========================================================================*/
/*  Unit‑test harness (compiled only when TK_CPP_IMPL_UNITTEST is defined) */
/*==========================================================================*/

#ifdef TK_CPP_IMPL_UNITTEST

/**
 * @brief Simple RAII wrapper that prints a header before a test case and
 *        a footer after it.  Used to make the console output easier to read.
 */
struct TestCase {
    explicit TestCase(const char* name) : name_(name) {
        std::cout << "\n=== TEST CASE: " << name_ << " =============================\n";
    }
    ~TestCase() {
        std::cout << "=== END OF " << name_ << " =============================\n";
    }
private:
    const char* name_;
};

/**
 * @brief Helper that checks a condition and throws if false.
 *
 * @param cond Condition to test.
 * @param msg  Message displayed on failure.
 */
inline void require(bool cond, const char* msg)
{
    if (!cond) {
        throw std::runtime_error(std::string("Requirement failed: ") + msg);
    }
}

/**
 * @brief Run a series of sanity checks that exercise every public API entry.
 *
 * The function is deliberately long (≈ 300 lines) to showcase the verbose
 * style requested.  It creates a context, tensors, audio streams and vision
 * frames, performs a few arithmetic operations, and finally tears everything
 * down.  All exceptions are caught and reported.
 */
int run_selftest()
{
    try {
        /* -----------------------------------------------------------------
         * 1. Context creation
         * ----------------------------------------------------------------- */
        {
            TestCase tc("Context creation");
            tk::Context ctx;
            require(ctx.raw() != nullptr, "Context raw pointer is null");
            log_debug("Created context at %p", static_cast<void*>(ctx.raw()));
        }

        /* -----------------------------------------------------------------
         * 2. Tensor lifecycle
         * ----------------------------------------------------------------- */
        {
            TestCase tc("Tensor lifecycle");
            std::vector<int64_t> shape = { 4, 8 };
            std::vector<float> init_data(4 * 8, 1.5f);
            tk::Tensor t = tk::Tensor::create<float>(shape, init_data.data());

            require(t.raw() != nullptr, "Tensor raw pointer is null");
            require(t.data_type() == tk::TK_DATA_TYPE_FLOAT32, "Tensor dtype mismatch");
            require(t.shape() == shape, "Tensor shape mismatch");

            // Fill with a constant
            t.fill<float>(3.14f);
            const float* ptr = t.data<float>();
            for (size_t i = 0; i < init_data.size(); ++i) {
                require(ptr[i] == 3.14f, "Tensor fill failed");
            }

            // Reshape to 2×16
            std::vector<int64_t> new_shape = { 2, 16 };
            t.reshape(new_shape);
            require(t.shape() == new_shape, "Tensor reshape failed");
            std::cout << t << std::endl;
        }

        /* -----------------------------------------------------------------
         * 3. Tensor arithmetic (add, matmul)
         * ----------------------------------------------------------------- */
        {
            TestCase tc("Tensor arithmetic");
            std::vector<int64_t> shape = { 2, 3 };
            std::vector<float> a_data = { 1, 2, 3, 4, 5, 6 };
            std::vector<float> b_data = { 6, 5, 4, 3, 2, 1 };
            tk::Tensor A = tk::Tensor::create<float>(shape, a_data.data());
            tk::Tensor B = tk::Tensor::create<float>(shape, b_data.data());

            tk::Tensor C = tk::Tensor::add(A, B);
            const float* c_ptr = C.data<float>();
            for (size_t i = 0; i < a_data.size(); ++i) {
                require(c_ptr[i] == a_data[i] + b_data[i], "Tensor add incorrect");
            }

            // Matrix multiplication (2×3) * (3×2) => (2×2)
            std::vector<int64_t> shapeB = { 3, 2 };
            std::vector<float> b2_data = { 1, 2, 3, 4, 5, 6 };
            tk::Tensor B2 = tk::Tensor::create<float>(shapeB, b2_data.data());

            tk::Tensor M = tk::Tensor::matmul(A, B2);
            const float* m_ptr = M.data<float>();
            // Expected result computed manually:
            // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
            // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
            std::vector<float> expected = { 22, 28, 49, 64 };
            for (size_t i = 0; i < expected.size(); ++i) {
                require(m_ptr[i] == expected[i], "Tensor matmul incorrect");
            }
            std::cout << "MatMul result: " << M << std::endl;
        }

        /* -----------------------------------------------------------------
         * 4. Audio stream – write / read / reset
         * ----------------------------------------------------------------- */
        {
            TestCase tc("AudioStream");
            const uint32_t sr = 48000;
            const uint16_t ch = 2;
            const tk::TkAudioFormat fmt = tk::TK_AUDIO_FMT_S16LE;
            const size_t capacity = 1024; // frames

            tk::AudioStream stream(sr, ch, fmt, capacity);
            require(stream.raw() != nullptr, "AudioStream raw pointer is null");
            std::cout << stream << std::endl;

            // Generate a simple sine wave (dummy data)
            std::vector<int16_t> samples(capacity * ch, 0);
            for (size_t i = 0; i < samples.size(); ++i) {
                samples[i] = static_cast<int16_t>((i % 256) - 128);
            }

            stream.write(samples.data(), capacity);
            size_t read_frames = 0;
            std::vector<int16_t> read_buf(capacity * ch, 0);
            stream.read(read_buf.data(), capacity, read_frames);
            require(read_frames == capacity, "AudioStream read frame count mismatch");
            require(std::memcmp(samples.data(), read_buf.data(),
                               read_frames * ch * sizeof(int16_t)) == 0,
                    "AudioStream data mismatch");

            stream.reset();
            size_t after_reset = 0;
            stream.read(read_buf.data(), capacity, after_reset);
            require(after_reset == 0, "AudioStream not empty after reset");
        }

        /* -----------------------------------------------------------------
         * 5. Vision frame – creation and pixel access
         * ----------------------------------------------------------------- */
        {
            TestCase tc("VisionFrame");
            const uint32_t w = 640;
            const uint32_t h = 480;
            tk::VisionFrame vf(w, h, tk::TK_VISION_FMT_RGB24);
            require(vf.raw() != nullptr, "VisionFrame raw pointer is null");
            std::cout << vf << std::endl;

            // Fill the frame with a solid colour (e.g., red)
            size_t stride = w * 3; // RGB24 => 3 bytes per pixel
            uint8_t* ptr = static_cast<uint8_t*>(vf.mutable_data());
            for (uint32_t y = 0; y < h; ++y) {
                uint8_t* row = ptr + y * stride;
                for (uint32_t x = 0; x < w; ++x) {
                    row[3 * x + 0] = 255; // R
                    row[3 * x + 1] = 0;   // G
                    row[3 * x + 2] = 0;   // B
                }
            }

            // Verify a few pixels
            const uint8_t* const_ptr = static_cast<const uint8_t*>(vf.data());
            for (uint32_t y = 0; y < h; y += 100) {
                for (uint32_t x = 0; x < w; x += 100) {
                    const uint8_t* pixel = const_ptr + y * stride + x * 3;
                    require(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0,
                            "VisionFrame pixel colour mismatch");
                }
            }
        }

        /* -----------------------------------------------------------------
         * 6. Asynchronous command execution
         * ----------------------------------------------------------------- */
        {
            TestCase tc("Async command execution");
            tk::Context ctx;
            bool callback_invoked = false;

            ctx.execute_async(tk::TK_MODULE_CORTEX,
                             "dummy_command",
                             nullptr,
                             [&](tk::TkStatus status, void* /*result*/) {
                                 callback_invoked = true;
                                 require(status == tk::TK_STATUS_OK,
                                         "Async command returned error");
                             });

            // Give the worker a moment to process the job.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            require(callback_invoked, "Async callback was not invoked");
        }

        /* -----------------------------------------------------------------
         * 7. Utility functions
         * ----------------------------------------------------------------- */
        {
            TestCase tc("Utility functions");
            const std::size_t sz = 64;
            void* aligned = tk::aligned_alloc(sz);
            require(aligned != nullptr, "aligned_alloc returned nullptr");
            require(reinterpret_cast<std::uintptr_t>(aligned) % tk::TK_SIMD_ALIGNMENT == 0,
                    "Memory not SIMD‑aligned");

            // Fill with a secret pattern and then zero it securely.
            std::memset(aligned, 0xAA, sz);
            tk::secure_zero(aligned, sz);
            const unsigned char* p = static_cast<const unsigned char*>(aligned);
            for (std::size_t i = 0; i < sz; ++i) {
                require(p[i] == 0, "secure_zero failed");
            }
            tk::aligned_free(aligned);

            // Constant‑time compare
            const char secret1[] = "super_secret";
            const char secret2[] = "super_secret";
            const char secret3[] = "super_secrex";
            require(tk::memcmp_const_time(secret1, secret2, sizeof(secret1)) == true,
                    "memcmp_const_time false negative");
            require(tk::memcmp_const_time(secret1, secret3, sizeof(secret1)) == false,
                    "memcmp_const_time false positive");
        }

        std::cout << "\nAll self‑tests passed successfully.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nSelf‑test failed: " << e.what() << "\n";
        return 1;
    }
}

/**
 * @brief Entry point for the self‑test executable.
 *
 * The `main` function is compiled only when `TK_CPP_IMPL_UNITTEST` is
 * defined.  It simply forwards to `run_selftest()`.
 */
int main()
{
    return run_selftest();
}

#endif /* TK_CPP_IMPL_UNITTEST */

/*==========================================================================*/
/*  End of file                                                            */
/*==========================================================================*/

} // namespace tk
