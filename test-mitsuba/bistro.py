import os
from typing import Callable
import mitsuba as mi
import drjit as dr
import time
import tqdm

mi.set_variant("cuda_ad_rgb")

os.makedirs("out/bistro", exist_ok=True)


def func(scene: mi.Scene, seed: mi.UInt32) -> mi.TensorXf:
    with dr.profile_range("render"):
        result = mi.render(scene, spp=1, seed=seed)
    return result


frozen = dr.freeze(func)


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
        if i >= b:
            duration -= time.time()
        dr.sync_thread()
        img = func(scene, mi.UInt32(i))
        dr.sync_thread()
        if i >= b:
            duration += time.time()
        images.append(img)
        mi.util.write_bitmap(f"out/bistro/{name}{i}.exr", img)
    return images, duration / n


scene = mi.load_file("data/bistro_singlebsdf/scene.xml")
n = 10
b = 3

print("Normal:")
ref, d_ref = run("normal", scene, func, n, b)
print("Frozen:")
res, d_res = run("frozen", scene, frozen, n, b)

assert frozen.n_recordings < n
for ref, res in zip(ref, res):
    assert dr.allclose(ref, res)

print(f"normal: {d_ref}s, frozen: {d_res}s")
