import tvm
import numpy as np

tgt="llvm -mcpu=core-avx2"

n = 64000
A = tvm.placeholder((n,), name='A')
C = tvm.compute(A.shape, lambda i: tvm.fast_exp(A[i]), name="C")
#C = tvm.compute(A.shape, lambda i: tvm.exp(A[i]), name="C")

s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=8)
#s[C].parallel(bx)
s[C].vectorize(tx)
# s[C].unroll(bx)
# s[C].unroll(tx)
fexp = tvm.build(s, [A, C], tgt, name="myexp")
print(tvm.lower(s, [A, C], simple_mode=True))
#print(fexp.get_source())
print(fexp.get_source("asm"))
ctx = tvm.context(tgt, 0)

a = tvm.nd.array(np.random.uniform(size=(n)).astype(A.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fexp(a, c)
#np.testing.assert_allclose(c.asnumpy(), np.exp(a.asnumpy()), atol=1e-3, rtol=1e-3)
ref = np.exp(a.asnumpy())
print(np.max(np.abs(c.asnumpy() - ref)))
# print(ref)
# print(c.asnumpy())
func = fexp
timer = func.time_evaluator(func.entry_name, ctx, number=1000)
print(timer(a, c).mean)
