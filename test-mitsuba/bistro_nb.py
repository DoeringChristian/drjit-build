# %% [markdown]
# # Frozen Functions
#
# In this book, you will learn how to use Dr.Jit's `freeze` decorator to improve
# the performance of rendering a complex scene.

# %%[markdown]
# First, we can import Mitsuba along with Dr.Jit and matplotlib. We also have to
# tell Mitsuba, which variant we want to use. Depending on your system, your GPU
# might not have enough VRAM. In that case, remove the "cuda_ad_rgb" variant.
# To record accurate kernel timings, we also enable Dr.Jit's kernel history.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import mitsuba as mi
import drjit as dr
import time
import tqdm

mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

dr.set_flag(dr.JitFlag.KernelHistory, True)

# %% [markdown]
# ## Rendering a Scene
#
# Let's define a function that takes a scene as well as a seed, and renders the
# scene with 1 sample per pixel. The seed can be either a Python `int`, or a Mitsuba
# `mi.UInt32`. When changing a Python `int` in the arguments of a frozen function,
# it would lead to the function being re-traced. Therefore, we use a `mi.UInt32` type.


# %%
def func(scene: mi.Scene, seed: mi.UInt32):
    return mi.render(scene, spp=1, seed=seed)


# %% [markdown]
# We can now load a complex scene. The bistro scene for example, contains many different
# shapes and materials which have to be traced when to compile the kernel.

# %%
scene = mi.load_file("data/bistro/scene.xml", resx=1920, resy=1080)

# %% [markdown]
# Rendering the scene, can be expensive, as the following code shows.
# We also measure the time, that the GPU spent executing kernels on the GPU, using
# Dr.Jit's kernel history. To get more accurate measurements, we render the scene 100 times.
# GPU frequencies can vary a lot, and skew the results, therefore it is advisable to lock the gpu frequencies.
# In some applications, part of the tracing cost can be hidden by kernel execution,
# if the code is designed in a way, that allows asynchronous execution on the GPU.
# We initialize the seed with `dr.opaque`, which will directly allocate and initialize
# memory on the GPU. When freezing the function, this will reduce the number of
# times the function has to be recorded before being able to replay it.

# %%

# Render the scene n times, to calculate a better estimate of the performance
n = 100
time_iter = 0
time_exec = 0
time_asm = 0
for i in tqdm.tqdm(range(n)):
    dr.kernel_history_clear()
    time_iter -= time.time() / n

    seed = dr.opaque(mi.UInt32, i)
    img = func(scene, seed)
    dr.eval(img)
    dr.sync_thread()

    time_iter += time.time() / n

    # Kernel assembly and execution time is stored in milliseconds
    history = dr.kernel_history()
    time_exec += (
        dr.sum([kernel["execution_time"] for kernel in history]) / 1000
    ) / n
    time_asm += (
        dr.sum(
            [
                kernel["codegen_time"]
                for kernel in history
                if kernel["type"] == dr.KernelType.JIT
            ]
        )
        / 1000
    ) / n


print(f"Rendering one frame of the bistro scene took {time_iter}s")
print(f"Assembling the kernels took {time_exec}s")
print(f"Executing the kernels took just {time_exec}s")

mi.Bitmap(img)

# %% [markdown]
# ## Creating a Frozen Function
#
# To use the frozen function decorator, we can either define a new function, that we annotate with `@dr.freeze`
# ```python
# @dr.freeze
# def frozen(scene: mi.Scene, seed: mi.UInt32):
#     return mi.render(scene, spp=1, seed=seed)
# ```
# or we can create a new frozen function instance from an existing function by calling


# %%
frozen = dr.freeze(func)


# %% [markdown]
# Multiple frozen functions can be created, referencing the same function. They
# will all have their own separate recording cache.
#
# Now we can call the frozen function, recording the kernels launched by `func`.
# The first time this will take longer than calling the function directly, since
# Dr.Jit has to traverse the inputs, record kernel calls, and construct outputs.

# %%
seed = dr.opaque(mi.UInt32, 0)

start = time.time()
img = frozen(scene, seed)
dr.sync_thread()
end = time.time()

print(f"Rendering the scene while recording the frozen function took {end - start}s")

mi.Bitmap(img)

# %% [markdown]
# We can check, that the function has been recorded, and that the recording is
# stored in the frozen function callable, using the `n_recordings` and `n_cached_recordings`
# properties.

# %%
assert frozen.n_recordings == 1
assert frozen.n_cached_recordings == 1


# %% [markdown]
# Subsequent calls to the function will be faster, since the kernels are simply replayed,
# after analyzing the inputs. Since we provided an opaque seed value, we are able
# to change it, without having to re-trace the function. If it was initialized as
# a literal (i.e. `seed = mi.UInt32(0)`), the second call would determine the literals
# that changed, and make them opaque automatically (auto opaque feature). Subsequent
# calls would then replay the function, without re-tracing it.

# %%
seed = dr.opaque(mi.UInt32, 1)

time_frozen_iter = 0
time_frozen_exec = 0

for i in tqdm.tqdm(range(n)):
    dr.kernel_history_clear()
    time_frozen_iter -= time.time() / n

    seed = dr.opaque(mi.UInt32, i)
    img = frozen(scene, seed)
    dr.eval(img)
    dr.sync_thread()

    time_frozen_iter += time.time() / n

    # Kernel Execution time is stored in milliseconds
    time_frozen_exec += (
        dr.sum([kernel["execution_time"] for kernel in dr.kernel_history()]) / 1000
    ) / n

print(f"Rendering the scene while replaying the function took {time_frozen_iter}s")
print(f"Executing the kernels took {time_frozen_exec}s")

mi.Bitmap(img)

# %% [markdown]
# To verify, that the function has not been re-traced, we can again use `n_recordings`,
# which gets incremented whenever the function is re-traced.

# %%
assert frozen.n_recordings == 1
assert frozen.n_cached_recordings == 1

# %% [markdown]
# ## Results
#
# Finally, we can plot a graph to visualize the performance gained by using frozen functions.

# %%
b = plt.bar(
    ["Normal", "Frozen"], [time_exec, time_frozen_exec], label="kernel", color = "C0"
)
plt.bar_label(b, fmt="{:0.3f}s", label_type = "center")
b = plt.bar([0], [time_asm], bottom=[time_exec], label="assembly", color="C2")
plt.bar_label(b, fmt = "{:0.3f}s", label_type = "center")
b = plt.bar(
    [0, 1],
    [time_iter - time_exec - time_asm, time_frozen_iter - time_frozen_exec],
    bottom=[time_exec + time_asm, time_frozen_exec],
    label="overhead",
    color="C1",
)
plt.bar_label(b, fmt="{:0.3f}s", label_type = "center")
plt.ylabel("Time in s")
plt.legend()
