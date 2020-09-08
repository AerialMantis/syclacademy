# SYCL Academy

## Exercise 17: Vector Load and Store
---

In this exercise you will learn how to use `vec` to explicitly vectorized your
kernel function and perform aligned vector loads and stores in order to compare
the performance difference.

---

### 1.) Use vectors

Now that global memory access is coalesced another optimization you can do is to
use the `vec` class to present the pixels in the image.

First create a `vec` from the r, g, b, a elements of the pixel and perform the
operations in the kernel function using the `vec` class.

### 1.) Align vector loads and stores

Once the kernel function is using `vec` to represent the pixels the next step is
to optimize the way the pixel data is read and written to global memory using
aligned loads and stores.

To do this use the `vec` member functions `load` and `store` to copy from
global memory to the `vec` and back again.

Make sure to calculate the offset to the position in global memory for the
pixels you are reading from or writing to.

Note that the `load` and `store` member functions take a `global_ptr` which can
be retrieved from an `accessor` by calling the `get_pointer` member function.

Compare the performance with aligned vector loads and stores against the
previous version.

## Build and execution hints

```
cmake -DSYCL_ACADEMY_USE_COMPUTECPP=ON -DSYCL_ACADEMY_INSTALL_ROOT=/insert/path/to/computecpp ..
make vector_load_and_store_source
./Code_Exercises/Exercise_17_Vector_Load_and_Store/vector_load_and_store_source
```
