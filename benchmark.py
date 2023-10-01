import numpy as np
import cupy as cp
import timeit

m1 = 0x5555555555555555
m2 = 0x3333333333333333
m4 = 0x0F0F0F0F0F0F0F0F
m8 = 0x00FF00FF00FF00FF
m16 = 0x0000FFFF0000FFFF
m32 = 0x00000000FFFFFFFF
h01 = 0x0101010101010101

f = np.vectorize(lambda x: bin(n).count("1"))

precalc_16bit = np.array(
    [bin(n).count("1") for n in range(2**16 - 1)], dtype=np.uint8
)


def py_builtin_bitcount(arr: np.ndarray):
    return f(arr)


def algo_bitcount(arr: np.ndarray):
    arr -= (arr >> 1) & m1
    arr = (arr & m2) + ((arr >> 2) & m2)
    arr = (arr + (arr >> 4)) & m4
    return (arr * h01) >> 56


def algo_bitcount_32bit(arr: np.ndarray):
    arr -= (arr >> 1) & m1
    arr = (arr & m2) + ((arr >> 2) & m2)
    arr = (arr + (arr >> 4)) & m4
    arr += arr >> 8
    arr += arr >> 16
    return arr & 0x7F


def precalc_bitcount_16bit(arr: np.ndarray):
    return precalc_16bit[arr]


n = 2 * 10**5
unpack_arr = (np.arange(n * 64) % 2).reshape(n, 64).astype(np.bool_)
arr = np.arange(n, dtype=np.uint64)
arr_sp = (np.minimum(np.arange(n * 4), 2**16 - 2)).reshape(n, 4).astype(np.uint16)

res = timeit.timeit("py_builtin_bitcount(arr)", globals=globals(), number=100)
print("vectorize:", res * 10, "ms")

res = timeit.timeit("unpack_arr.sum(axis=1)", globals=globals(), number=100)
print("unpack:", res * 10, "ms")

res = timeit.timeit("algo_bitcount(arr)", globals=globals(), number=100)
print("algo:", res * 10, "ms")

res = timeit.timeit(
    "precalc_bitcount_16bit(arr_sp).sum(axis=1)", globals=globals(), number=100
)
print("precalc:", res * 10, "ms")


# res = timeit.timeit("np.bitwise_count(arr)", globals=globals(), number=100)
# print("buildin:", res * 10, "ms")


x = cp.asarray(arr)


def cp_algo_bitcount(arr: cp.ndarray):
    arr -= (arr >> 1) & m1
    arr = (arr & m2) + ((arr >> 2) & m2)
    arr = (arr + (arr >> 4)) & m4
    return (arr * h01) >> 56


res = timeit.timeit("cp_algo_bitcount(x)", globals=globals(), number=100)
print("gpu_algo:", res * 10, "ms")

cp_kernel_fusion_bitcount = cp.ElementwiseKernel(
    "uint64 x",
    "uint64 z",
    """
   z = (x & 0x5555555555555555) + (x >> 1 & 0x5555555555555555);
   z = (z & 0x3333333333333333) + (z >> 2 & 0x3333333333333333);
   z = (z & 0x0F0F0F0F0F0F0F0F) + (z >> 4 & 0x0F0F0F0F0F0F0F0F);
   z = (z & 0x00FF00FF00FF00FF) + (z >> 8 & 0x00FF00FF00FF00FF);
   z = (z & 0x0000FFFF0000FFFF) + (z >> 16 & 0x0000FFFF0000FFFF);
   z = (z & 0x00000000FFFFFFFF) + (z >> 32 & 0x00000000FFFFFFFF);
   """,
    "pros_64bit",
)

res = timeit.timeit("cp_kernel_fusion_bitcount(x)", globals=globals(), number=100)
print("gpu_kernel_fusion_algo:", res * 10, "ms")


"""
codespaces上での実行結果
unpack: 48.809579450003184 ms
vectorize: 6.160439129998849 ms
algo: 0.9873942499962141 ms
precalc: 4.742630679998001 ms
buildin: 0.09576133999871672 ms
"""
