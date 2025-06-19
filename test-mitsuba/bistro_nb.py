# %% [markdown]
"""
# Frozen Functions

"""

# %%
import matplotlib.pyplot as plt
import mitsuba as mi
import drjit as dr
import time

mi.set_variant("cuda_ad_rgb")

dr.set_flag(dr.JitFlag.KernelHistory, True)

# %% [markdown]
"""
Let's define a function that takes a scene as well as a seed, and renders the scene with 1 sample per pixel.
"""


# %%
def func(scene: mi.Scene, seed: mi.UInt32):
    return mi.render(scene, spp=1, seed=seed)


# %% [markdown]
"""
We can now load a complex scene. The bistro scene for example, contains many different
BSDFs, which will all have to be traced when not using frozen functions.
"""
# %%
scene = mi.load_file("data/bistro/scene.xml")

# %% [markdown]
"""
Rendering the scene, can be expensive, as the following code shows.
A lot of time is spent tracing functions for scene elements.
"""

# %%
seed = dr.opaque(mi.UInt32, 0)

dr.kernel_history_clear()
start = time.time()
img = func(scene, seed)
dr.sync_thread()
end = time.time()

duration = end - start
# Kernel Execution time is stored in milliseconds
execution_time = (
    dr.sum([kernel["execution_time"] for kernel in dr.kernel_history()]) / 1000
)


print(f"Rendering the bistro scene took {duration}s")

plt.imshow(mi.util.convert_to_bitmap(img))
plt.axis("off")

# %% [markdown]
"""
We can actually check how long it took to execute the kernels.
"""

# %%

print(f"Executing the kernels took just {execution_time}s")


# %%
@dr.freeze
def frozen(scene: mi.Scene, seed: mi.UInt32):
    return mi.render(scene, spp=1, seed=seed)


# %% [markdown]
"""
The first call to the frozen function will take longer than calling the function directly.
Dr.Jit has to traverse the inputs, record kernel calls, and construct outputs.
"""

# %%
seed = dr.opaque(mi.UInt32, 0)

start = time.time()
img = frozen(scene, seed)
dr.sync_thread()
end = time.time()

print(f"Rendering the scene while recording the frozen function took {end - start}s")

plt.imshow(mi.util.convert_to_bitmap(img))
plt.axis("off")


# %% [markdown]
"""
Subsequent calls to the function will be faster, since the kernels are simply replayed,
after analyzing the inputs.
"""

# %%
seed = dr.opaque(mi.UInt32, 1)

dr.kernel_history_clear()
start = time.time()
img = frozen(scene, seed)
dr.sync_thread()
end = time.time()

duration_frozen = end - start

# Kernel Execution time is stored in milliseconds
execution_time_frozen = (
    dr.sum([kernel["execution_time"] for kernel in dr.kernel_history()]) / 1000
)

print(f"Rendering the scene while replaying the function took {duration_frozen}s")

plt.imshow(mi.util.convert_to_bitmap(img))
plt.axis("off")

print(f"Executing the kernels took {execution_time_frozen}s")

# %% [markdown]
"""
Finally, we can plot a graph to visualize the performance gained from using frozen functions.
"""

# %%
plt.bar([0, 1], [execution_time, execution_time_frozen])
plt.bar(
    [0, 1],
    [duration - execution_time, duration_frozen - execution_time_frozen],
    bottom=[execution_time, execution_time_frozen],
)
