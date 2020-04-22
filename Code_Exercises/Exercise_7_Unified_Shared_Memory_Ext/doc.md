# Application Development with SYCL

### Exercise 6: Unified Shared Memory (extension)

---

In this first exercise you will learn:
* How to allocate memory a region of unified shared memory on a device.
* How to perform copy and fill operations on unified shared memory regions.
* How to enqueue SYCL kernels using unified shared memory.

---

The [Intel extension for unified shared memory](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc) (USM) adds support for allocating and manipulating regions of unified shared memory in SYCL for devices which support it. This adds a new distinct memory model to SYCL along side the existing one which uses buffers and accessors.

There are various levels of USM which can be supported by a device, which provide varying levels of functionality. In this exercise we are going to focus on `Explicit USM`, which allows you to allocate a persistent pointer to a region of USM on the host or on a device and copy to and from it.

However, USM does not provide any implicit data movement or data dependency analysis which you would ordinarily get with `accessors`, so any copies must be done manually by calling `memcpy` and dependencies must be manually defined using `events`.

1.) Check for the USM extension

The first thing you'll need to do when wanting to use USM is to check whether the `device` you're running on supports it. You can do this can querying the `device` for the extension `"cl_intel_unified_shared_memory_preview"`.

2.) Allocate USM memory on the device

Once you have identified that the `device` supports USM you can go ahead and allocate memory on the device. To do this you call `experimental::malloc_device`, which takes a template parameter specifying the type you are allocating, and then as function parameters the size in elements and the `queue` you wish to allocate with. Note the `queue` you specify must be associated with the `device` you wish to allocate on.

Do this for both inputs and the output.

3.) Copy the inputs

Next you'll need to copy your input data to the USM memory regions you allocated for the inputs. You can do this by calling the `queue` meber function `memcpy`, which takes a destination pointer, source pointer and a size in bytes.

Each of these will return an `event`, you should store this event to wait on it later.

Do this for both inputs.

4.) Initialize the output

Next you'll need to zero the values at the USM memory region you allocated for the output. You can do this by calling the `queue` member function `fill`, which takes pointer, a value and a size in elements.

Again you should store teh `event` returned so you can wait on it later.

4a.) Construct `use_wrapper` objects (additional step required for ComputeCpp)

If you are using ComputeCpp you must follow this additional step, otherwise you can ignore it and move on.

You must wrap each of the pointers you have allocated in a `use_wrapper` object, you do this by simply constructing a `use_wrapper` and passing the pointer to the constructor.

Once you have done this move onto step 5, using these `use_wrapper` objects in place of the raw pointers.

5.) Wait on the `copy`/`fill` to complete

Before you enqueue the kernel itself you must wait for the `memcpy` and `fill` operations to complete.

You can do this by constructing a `std::vector<event>` with the `events` from each of the `memcpy` and `fill` operations and then call the static function `event::wait` on it.

6.) Enqueue a kernel

Now that all the memory is allocated, copies and initialized, you can enqueue a kernel to perform the computation.

When using USM there shortcut member functions for the `queue` class that enqueue kernel functions directly without having to create a command group, since you don't need to create accessors.

To launch a kernel call the `parallel_for` member function of the `queue` and capture the pointers you have allocated (the `use_wrapper` objects in the case of ComputeCpp).

In the kernel itself use write a vector add operation that takes the value at the index of each input pointers, add them together and assign the result to the value at the index of the output pointer.

The `parallel_for` function will also return an `event` so you should wait on that to wait for the kernel to complete before continuing.

7.) Copy the result back

Once the kernel has completed you can copy the result back, similarly to the inputs you can do this by calling the `queue` member function `memcpy`, and waiting on the `event` that is returned.
