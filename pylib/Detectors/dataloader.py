import json
import struct
import numpy as np

def read_im(stdin):
    buf = stdin.read(16)
    if not buf:
        return None
    (l, width, height, batchsize) = struct.unpack('>IIII', buf)
    buf = stdin.read(l)
    return np.frombuffer(buf, dtype='uint8').reshape((batchsize, height, width, 3))