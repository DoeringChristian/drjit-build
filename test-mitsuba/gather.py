import drjit as dr
from drjit.auto import Float, UInt32, ArrayXu, ArrayXf

dr.set_flag(dr.JitFlag.KernelHistory, True)

n = 2**24
p = 5
m = 1024

source = dr.rand(Float, n * p)
dr.eval(source)

index = UInt32(dr.rand(Float, m) * 1024)

y = dr.gather(ArrayXf, source, index, shape = (p, m))

dr.kernel_history_clear()
dr.eval(y)
history = dr.kernel_history((dr.KernelType.JIT,))

for k in history:
    print(f"Launched Kernel {k['hash']}")
