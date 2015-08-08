import numpy
from numpy.linalg import norm
import reikna.cluda as cluda
from reikna.linalg import MatrixMul
import pyopencl as cl

def main2():
    N = 256

    api = cluda.ocl_api()
    thr = api.Thread.create()

    program = thr.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM float *dest,
        GLOBAL_MEM float *a,
        GLOBAL_MEM float *b)
    {
      const SIZE_T i = get_local_id(0);
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = program.multiply_them

    a = numpy.random.randn(N).astype(numpy.float32)
    b = numpy.random.randn(N).astype(numpy.float32)
    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    dest_dev = thr.empty_like(a_dev)

    for i in xrange(100000):
        #res = a_dev * b_dev
        multiply_them(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    #print dest_dev.get()

    print((dest_dev.get() - a * b == 0).all())


def main():
    api = cluda.ocl_api()
    thr = api.Thread.create()
    print thr
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
    print res_reference
    #print(norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6)

platform = cl.get_platforms()
print platform
my_devices = platform[0].get_devices()
print my_devices

main2()
