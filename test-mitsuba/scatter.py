import drjit as dr
from drjit.auto import Float, Float16, UInt32, ArrayXu, ArrayXf, ArrayXf16

dr.set_flag(dr.JitFlag.KernelHistory, True)

n = 2**24
p = 5
m = 1024

target = dr.zeros(Float16, n * p)
dr.eval(target)

index = UInt32(dr.rand(Float, m) * 1024)
value = dr.rand(ArrayXf16, (p, m))

dr.eval(value)

dr.scatter_add(target, value, index)

dr.kernel_history_clear()
dr.eval(target)
history = dr.kernel_history((dr.KernelType.JIT,))

for k in history:
    print(f"Launched Kernel {k['hash']}")
