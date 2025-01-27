import os
from typing import Callable
import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

os.makedirs("out/bistro", exist_ok = True)


def func(scene: mi.Scene, seed: mi.UInt32) -> mi.TensorXf:
    with dr.profile_range("render"):
        result = mi.render(scene, spp=1, seed=seed)
    return result


frozen = dr.freeze(func)


def run(name: str, scene: mi.Scene, n: int, func: Callable[[mi.Scene, mi.UInt32], mi.TensorXf])->list[mi.TensorXf]:
    images = []
    for i in range(n):
        img = func(scene, mi.UInt32(i))
        images.append(img)
        mi.util.write_bitmap(f"out/bistro/{name}{i}.exr", img)
    return images

scene = mi.load_file("data/bistro_singlebsdf/scene.xml")
n = 10

ref = run("normal", scene, n, func)
res = run("frozen", scene, n, frozen)

assert frozen.n_recordings < n
for ref, res in zip(ref, res):
    assert dr.allclose(ref, res)
