import numpy as np

m1 = 0x5555555555555555
m2 = 0x3333333333333333
m4 = 0x0F0F0F0F0F0F0F0F
m8 = 0x00FF00FF00FF00FF
m16 = 0x0000FFFF0000FFFF
m32 = 0x00000000FFFFFFFF
h01 = 0x0101010101010101

f = np.vectorize(lambda x: bin(x).count("1"))

precalc_16bit = np.array(
    [bin(n).count("1") for n in range(2**16 - 1)], dtype=np.uint8
)


def py_builtin_bitcount(arr: np.ndarray):
    return f(arr)


def algo_bitcount(arr: np.ndarray):
    arr = (arr & 0x5555555555555555) + (arr >> 1 & 0x5555555555555555)
    arr = (arr & 0x3333333333333333) + (arr >> 2 & 0x3333333333333333)
    arr = (arr & 0x0F0F0F0F0F0F0F0F) + (arr >> 4 & 0x0F0F0F0F0F0F0F0F)
    arr = (arr & 0x00FF00FF00FF00FF) + (arr >> 8 & 0x00FF00FF00FF00FF)
    arr = (arr & 0x0000FFFF0000FFFF) + (arr >> 16 & 0x0000FFFF0000FFFF)
    return (arr & 0x00000000FFFFFFFF) + (arr >> 32 & 0x00000000FFFFFFFF)


def algo_bitcount2(arr: np.ndarray):
    arr -= (arr >> 1) & m1
    arr = (arr & m2) + ((arr >> 2) & m2)
    arr = (arr + (arr >> 4)) & m4
    arr += arr >> 8
    arr += arr >> 16
    arr += arr >> 32
    return arr & 0x7F


def algo_bitcount3(arr: np.ndarray):
    arr -= (arr >> 1) & m1
    arr = (arr & m2) + ((arr >> 2) & m2)
    arr = (arr + (arr >> 4)) & m4
    return (arr * h01) >> 56


def better_bitcount_32bit(arr: np.ndarray):
    arr -= (arr >> 1) & 0x55555555
    arr = (arr & 0x33333333) + ((arr >> 2) & 0x33333333)
    arr = (arr + (arr >> 4)) & 0x0F0F0F0F
    arr += arr >> 8
    arr += arr >> 16
    return arr & 0x7F


def precalc_bitcount_16bit(arr: np.ndarray):
    return precalc_16bit[arr]
