import ctypes
from ctypes import cdll
import numpy as np
import time

lib = cdll.LoadLibrary('./build/ssd.so')
ssd_cuda = lib.ssd


def compute_ssd(window, target, out, window_width, window_height,
                target_width, target_height, out_width, out_height):
    ssd_cuda(ctypes.c_void_p(window.ctypes.data),
             ctypes.c_void_p(target.ctypes.data),
             ctypes.c_void_p(out.ctypes.data),
             ctypes.c_int32(window_width), ctypes.c_int32(window_height),
             ctypes.c_int32(target_width), ctypes.c_int32(target_height),
             ctypes.c_int32(out_width), ctypes.c_int32(out_height))


window_width = 100
window_height = 100
target_width = 120
target_height = 106
out_width = 21
out_height = 7

window = np.ones((window_height, window_width), dtype='int32')
target = np.zeros((target_height, target_width), dtype='int32')
out = np.zeros((out_height, out_width), dtype='int32')

compute_ssd(window, target, out, window_width, window_height,
            target_width, target_height, out_width, out_height)
tic = time.time()
for i in range(5):
    compute_ssd(window, target, out, window_width, window_height,
                target_width, target_height, out_width, out_height)
print(time.time() - tic)

out_np = np.zeros((out_height, out_width), dtype='int32')
tic = time.time()
for i in range(5):
    for i in range(out_height):
        for j in range(out_width):
            diff = target[i: i + window.shape[0], j: j + window.shape[1]] - window
            out_np[i, j] = np.sum(np.multiply(diff, diff).reshape(-1))
print(time.time() - tic)

print(out - out_np)

