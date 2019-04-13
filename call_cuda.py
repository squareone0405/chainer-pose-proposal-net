import ctypes
from ctypes import cdll
import numpy as np
import time

lib = cdll.LoadLibrary('./build/ssd.so')
ssd_cuda = lib.ssd


def compute_ssd(window, target, out, window_width, window_height,
                target_width, target_height, out_width, out_height, num):
    ssd_cuda(ctypes.c_void_p(window.ctypes.data),
             ctypes.c_void_p(target.ctypes.data),
             ctypes.c_void_p(out.ctypes.data),
             ctypes.c_void_p(window_width.ctypes.data), 
             ctypes.c_void_p(window_height.ctypes.data), 
             ctypes.c_void_p(target_width.ctypes.data), 
             ctypes.c_void_p(target_height.ctypes.data), 
             ctypes.c_int32(out_width), ctypes.c_int32(out_height), ctypes.c_int32(num))


window_width = np.array([50, 50], dtype='int32')
window_height = np.array([50, 50], dtype='int32')
target_width = np.array([70, 70], dtype='int32')
target_height = np.array([60, 60], dtype='int32')
out_width = 21
out_height = 11
num = 2

window = np.ones((window_height[0], window_width[0] * num), dtype='int32')
target = np.zeros((target_height[0], target_width[0] * num), dtype='int32')
out = np.zeros((out_height, out_width * num), dtype='int32')

compute_ssd(window, target, out, window_width, window_height,
            target_width, target_height, out_width, out_height, num)
tic = time.time()
for i in range(5):
    compute_ssd(window, target, out, window_width, window_height,
                target_width, target_height, out_width, out_height, num)
print(time.time() - tic)

out_np = np.zeros((out_height, out_width), dtype='int32')
tic = time.time()
for i in range(5):
    for i in range(out_height):
        for j in range(out_width):
            diff = target[i: i + window.shape[0], j: j + window.shape[1]] - window
            out_np[i, j] = np.sum(np.multiply(diff, diff).reshape(-1))
print(time.time() - tic)

print(out)
print(out_np)
print(out.shape)
print(out_np.shape)

