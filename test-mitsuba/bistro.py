import os
from typing import Callable
import mitsuba as mi
import drjit as dr
import time
import tqdm

mi.set_variant("cuda_ad_rgb")

# dr.set_flag(dr.JitFlag.LaunchBlocking, True)
dr.set_log_level(dr.LogLevel.Warn)

os.makedirs("out/bistro", exist_ok=True)


def func(scene: mi.Scene, seed: mi.UInt32) -> mi.TensorXf:
    with dr.profile_range("render"):
        result = mi.render(scene, spp=1, seed=seed)
    return result


frozen = dr.freeze(func, auto_opaque=True)


def run(
    name: str,
    scene: mi.Scene,
    func: Callable[[mi.Scene, mi.UInt32], mi.TensorXf],
    n: int,
    b: int = 3,
) -> list[mi.TensorXf]:
    images = []
    duration = 0
    for i in tqdm.tqdm(range(b + n)):
        with dr.profile_range("iteration"):
            if i >= b:
                duration -= time.time()
            dr.sync_thread()
            seed = mi.UInt32(i)
            # dr.make_opaque(seed)
            img = func(scene, seed)
            dr.sync_thread()
            if i >= b:
                duration += time.time()
            images.append(img)
            mi.util.write_bitmap(f"out/bistro/{name}{i}.exr", img)
    return images, duration / n


scene = mi.load_file("data/bistro_singlebsdf/scene.xml", parallel=False)
n = 100
b = 10

print("Normal:")
ref, d_ref = run("normal", scene, func, n, b)
print("Frozen:")
res, d_res = run("frozen", scene, frozen, n, b)

assert frozen.n_recordings < n
for ref, res in zip(ref, res):
    assert dr.allclose(ref, res)

print(f"normal: {d_ref}s, frozen: {d_res}s")
print(f"{frozen.n_recordings=}")
