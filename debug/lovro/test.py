from typing import Union, Tuple, List, Any, Optional, Sequence

import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("cuda_ad_rgb")


SPP = 2048
SPP_OBO = 512
NUM_ITERS = 100
H, W = 256, 256

np.random.seed(2)

random_elements = ["white", "green", "red"]



def generate_scene():
    scene_dict = mi.cornell_box()
    return scene_dict


ref_scene_dict = mi.cornell_box()
ref_scene = mi.load_dict(ref_scene_dict)

integrator_prb = mi.load_dict(
    {
        "type": "prb",
    }
)

ref_img = mi.render(ref_scene, integrator=integrator_prb, spp=SPP)
dr.eval(ref_img)

import os

# import pyexr
try:
    os.mkdir("out")
except FileExistsError:
    pass

optimizer_bsdf = mi.ad.Adam(lr=0.0001)

optimization_scene_dict = generate_scene()
optimization_scene = mi.load_dict(optimization_scene_dict)
optimization_scene_params = mi.traverse(optimization_scene)
optimization_scene_params.keep(['white.reflectance.value', 'green.reflectance.value', 'red.reflectance.value'])
print(optimization_scene_params)

for random_element in random_elements:
    optimizer_bsdf[f"{random_element}.reflectance.value"] = optimization_scene_params[
        f"{random_element}.reflectance.value"
    ]
print(optimizer_bsdf)
optimization_scene_params.update(optimizer_bsdf)

for it in range(NUM_ITERS):

    optimization_scene_params.update(optimizer_bsdf)

    img_prb = mi.render(
        optimization_scene,
        optimization_scene_params,
        integrator=integrator_prb,
        spp=SPP_OBO,
        seed=3,
    )
    loss = dr.sum(dr.square(img_prb - ref_img))
    dr.backward(loss)
    optimizer_bsdf.step()

    for random_element in random_elements:
        optimizer_bsdf[f"{random_element}.reflectance.value"] = dr.clip(
            optimizer_bsdf[f"{random_element}.reflectance.value"], 1e-5, 1.0
        )

    print(f"Iter {it}: Loss = {loss.item()}")
    mi.util.write_bitmap(f"out/iter_{it:05d}_img.png", img_prb)
