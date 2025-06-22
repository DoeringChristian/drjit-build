from typing import Literal, Type
import drjit as dr
import drjit.nn as nn
import drjit.hashgrid as hg
from drjit.opt import Adam, GradScaler
import mitsuba as mi
import matplotlib.pyplot as plt
import tqdm

mi.set_variant("cuda_ad_rgb")

from drjit.auto.ad import (
    Float,
    Float16,
    Float32,
    Float64,
    UInt32,
    Bool,
    TensorXf,
    TensorXf16,
    ArrayXf,
    ArrayXf16
)


def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.square(pdf_a)
    b2 = dr.square(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))


class GeometrySampler:
    def __init__(self, scene: mi.Scene) -> None:
        self.sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        self.scene = scene

        area = []
        for shape in scene.shapes():
            if not shape.is_emitter():
                area.append(shape.surface_area()[0])
            else:
                area.append(0.0)

        area = Float(area)
        area /= dr.sum(area)

        self.shape_sampler = mi.DiscreteDistribution(area)

    def seed(self, seed, n):
        self.sampler.seed(seed, n)

    def sample(self, active: bool | Bool = True) -> mi.SurfaceInteraction3f:
        shape_index = self.shape_sampler.sample(self.sampler.next_1d(), active)
        shape: mi.ShapePtr = dr.gather(
            mi.ShapePtr, self.scene.shapes_dr(), shape_index, active
        )

        ps: mi.PositionSample3f = shape.sample_position(0.0, self.sampler.next_2d(), active)
        si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
        si.shape = shape
        bsdf = shape.bsdf()

        sample = self.sampler.next_2d()
        active_two_sided = mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
        si.wi = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sample),
            mi.warp.square_to_uniform_hemisphere(sample),
        )

        return si

class Model:
    def __init__(self, scene: mi.Scene, width: int = 32, hidden: int = 2) -> None:
        self.scene = scene

        self.encoding = hg.HashGridEncoding(3, n_levels=16, n_features_per_level=2)

        n_input_features = self.encoding.out_features  # encoding
        n_input_features += 3  # p
        n_input_features += 3  # w

        sequential = [
            nn.Linear(n_input_features, width, True),
            nn.ReLU(),
        ]
        for _ in range(hidden):
            sequential.append(nn.Linear(width, width, True))
            sequential.append(nn.LeakyReLU())

        sequential.append(nn.Linear(width, 3, True))

        self.mlp = nn.Sequential(
            *sequential
        )

    def alloc(self, dtype: Type[dr.ArrayBase]):
        self.encoding = self.encoding.alloc(dtype)
        self.mlp = self.mlp.alloc(dtype)

    def pack(self, layout: Literal["inference", "training"] = "inference"):
        self.weights, self.mlp = nn.pack(self.mlp, layout)

    def __call__(
        self, si: mi.SurfaceInteraction3f, active: bool | Bool = True
    ) -> mi.Spectrum:

        bbox = self.scene.bbox()
        p = (si.p - bbox.min) / (bbox.max - bbox.min)

        features = self.encoding(p)

        features = ArrayXf16(features)
        p = ArrayXf16(p)
        wi = ArrayXf16(si.wi)

        features = ArrayXf16(*features, *p, *wi)

        output = mi.Spectrum(self.mlp(nn.CoopVec(features)))

        return output


def render_rhs(
    it, scene: mi.Scene, model: Model, si_lhs: mi.SurfaceInteraction3f, batch_size: int, M: int
):
    with dr.suspend_grad():
        sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        sampler.seed(it, batch_size * M)

        indices = dr.arange(mi.UInt, 0, batch_size)
        indices = dr.repeat(indices, M)
        si_lhs = dr.gather(type(si_lhs), si_lhs, indices)

        L = mi.Spectrum(0)
        ctx = mi.BSDFContext()

        # Emitter sampling

        bsdf = si_lhs.bsdf()

        ds, em_weight = scene.sample_emitter_direction(si_lhs, sampler.next_2d(), True)
        wo = si_lhs.to_local(ds.d)

        bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(ctx, si_lhs, wo)

        mis_em = mis_weight(ds.pdf, bsdf_pdf_em)

        L += mis_em * bsdf_val_em * em_weight

        # BSDF sampling

        bsdf_sample, bsdf_weight = bsdf.sample(ctx, si_lhs, sampler.next_1d(), sampler.next_2d())

        ray = si_lhs.spawn_ray(si_lhs.to_world(bsdf_sample.wo))

        si_rhs: mi.SurfaceInteraction3f = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=False)

        ds = mi.DirectionSample3f(scene, si=si_rhs, ref=si_lhs)

        mis_bsdf = mis_weight(bsdf_sample.pdf, scene.pdf_emitter_direction(si_lhs, ds))

        L += mis_bsdf * bsdf_weight * model(si_rhs)

        L = dr.block_sum(L, M) / M

        return L

class Integrator(mi.SamplingIntegrator):
    def __init__(self, model: Model) -> None:
        super().__init__(mi.Properties())

        self.model = model

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium | None = None,
        active: Bool = True,
    ) -> tuple[mi.Spectrum, Bool, list[Float]]:
        self.model.pack("inference")

        si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)

        # si, f, _ = self.first_non_specular_or_null_si(scene, si, sampler)

        Le = si.emitter(scene).eval(si)

        # discard the null bsdf backside
        # null_face = ~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide) & (
        #     si.wi.z < 0
        # )
        # mask = si.is_valid() & ~null_face

        L_n = dr.detach(self.model(si, active))

        L = Le + L_n

        # L = render_rhs(0, scene, self.model, si, dr.width(si.p), 32)

        return L, si.is_valid(), []


scene: mi.Scene = mi.load_dict(mi.cornell_box())

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(si.p.x, si.p.y, si.p.z)
# plt.show()

model = Model(scene)

model.alloc(TensorXf16)

integrator = Integrator(model)
img = integrator.render(scene, sensor = scene.sensors()[0], seed = 0, spp = 1)

# plt.imshow(img)
# plt.show()

def train():
    iterations = 1_000
    batch_size = 1024
    M = 32

    model.pack("training")

    # Optimize a single-precision copy of the parameters
    opt = Adam(
        lr=1e-3,
        params={
            "encoding": Float32(model.encoding.data),
            "mlp": Float32(model.weights),
        },
    )

    gsampler = GeometrySampler(scene)

    # This is an adaptive mixed-precision (AMP) optimization, where a half
    # precision computation runs within a larger single-precision program.
    # Gradient scaling is required to make this numerically well-behaved.
    scaler = GradScaler()

    iterator = tqdm.tqdm(range(iterations))

    for it in iterator:
        gsampler.seed(it, batch_size)

        model.weights[:] = Float16(opt["mlp"])
        model.encoding.data[:] = Float16(opt["encoding"])

        si_lhs = gsampler.sample()

        L_lhs = model(si_lhs)

        L_rhs = render_rhs(it, scene, model, si_lhs, batch_size, M)

        loss = dr.square(L_lhs - L_rhs)
        loss = dr.mean(loss, axis = None)
        print(loss )

        dr.backward(scaler.scale(loss))

        scaler.step(opt)

        # Eval

        loss = loss[0]
        iterator.set_postfix({"loss": f"{loss:.8f}"})

train()

integrator = Integrator(model)
img = integrator.render(scene, sensor = scene.sensors()[0], seed = 0, spp = 1)

plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
