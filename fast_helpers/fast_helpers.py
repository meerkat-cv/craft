import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL('/code/libs/craft/fast_helpers/build/libfast_helpers.so')

c_find_boxes = lib.find_boxes
c_find_boxes.argtypes = [
    ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),
]

def find_boxes(markers, out_boxes):
    c_find_boxes(markers, markers.shape[1], markers.shape[0], out_boxes.shape[0], out_boxes)

# markers = np.zeros((20, 20), dtype=np.int32)
# out_boxes = np.zeros(20, dtype=np.int32)
# find_boxes(markers, out_boxes)
# print('OutBoxes', out_boxes)
