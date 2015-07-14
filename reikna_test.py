import numpy
from numpy.linalg import norm
import reikna.cluda as cluda
from reikna.linalg import MatrixMul
import pyopencl as cl

def main():
    platform = cl.get_platforms()
    print platform
    my_devices = platform[0].get_devices(device_type = cl.devices_type.ALL)
    print my_devices
    return
    api = cluda.ocl_api()
    thr = api.Thread.create()

    shape1 = (100, 200)
    shape2 = (200, 100)

    a = numpy.random.randn(*shape1).astype(numpy.float32)
    b = numpy.random.randn(*shape2).astype(numpy.float32)
    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    res_dev = thr.array((shape1[0], shape2[1]), dtype=numpy.float32)

    dot = MatrixMul(a_dev, b_dev, out_arr=res_dev)
    dotc = dot.compile(thr)
    dotc(res_dev, a_dev, b_dev)

    res_reference = numpy.dot(a, b)

    print(norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6)

main()
