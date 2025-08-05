from tqdm.auto import tqdm
import imageio.v3 as iio
import drjit as dr
import drjit.nn as nn
from drjit.opt import Adam, GradScaler
from drjit.auto.ad import Texture2f, TensorXf, TensorXf16, Float16, Float32, Array2f, Array3f

# Load a test image and construct a texture object
ref = TensorXf(iio.imread("https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/wave-128.png") / 256)
tex = Texture2f(ref)

# Create a two dimensional hash grid encoding, with 8 levels, 2 features per
# level and a scaling factor between levels of 1.5.
enc = nn.HashGridEncoding(
    Float16,
    2,
    n_levels=8,
    n_features_per_level=2,
    per_level_scale=1.5,
)

# Alternatively we can also use a permutohedral encoding. In contrast to a hash
# grid, it uses triangles, tetrahedrons and their higher dimensional
# equivalences as simplexes. Their vertex count scales linearly with dimension,
# allowing for higher dimensional inputs, while keeping the memory lookup
# overhead minimal.
# Uncomment the following lines to enable the permutohedral encoding.
# enc = nn.PermutoEncoding(
#     Float16,
#     2,
#     n_levels=8,
#     n_features_per_level=2,
#     per_level_scale=1.5,
# )
print(enc)


# Establish the network structure.
# In contrast to the previous example, we use a HashEncodingLayer, referencing
# the previously created hash grid. Its parameters will not be part of the
# packed weights, and have to be handled separately.
net = nn.Sequential(
    nn.HashEncodingLayer(enc),
    nn.Cast(Float16),
    nn.Linear(-1, -1, bias=False),
    nn.LeakyReLU(),
    nn.Linear(-1, -1, bias=False),
    nn.LeakyReLU(),
    nn.Linear(-1, -1, bias=False),
    nn.LeakyReLU(),
    nn.Linear(-1, 3, bias=False),
    nn.Exp()
)

# Instantiate the network for a specific backend + input size
net = net.alloc(TensorXf16, 2)

# Convert to training-optimal layout
weights, net = nn.pack(net, layout='training')
print(net)

# Optimize a single-precision copy of the parameters.
# In addition to the network weights, we also add the parameters of the
# encoding.
opt = Adam(
    lr=1e-3,
    params={
        "mlp.weights": Float32(weights),
        "enc.params": Float32(enc.params),
    },
)

# This is an adaptive mixed-precision (AMP) optimization, where a half
# precision computation runs within a larger single-precision program.
# Gradient scaling is required to make this numerically well-behaved.
scaler = GradScaler()

res = 256

rng = dr.rng()
for i in tqdm(range(40000)):
    # Update network state from optimizer
    weights[:] = Float16(opt['mlp.weights'])
    # Update the encoding parameters as well
    enc.params[:] = Float16(opt['enc.params'])

    # Generate jittered positions on [0, 1]^2
    t = dr.arange(Float32, res)
    p = (Array2f(dr.meshgrid(t, t)) + rng.random(Array2f, (2, res * res))) / res

    # Evaluate neural net + L2 loss
    img = Array3f(net(nn.CoopVec(p)))
    loss = dr.squared_norm(tex.eval(p) - img)

    # Mixed-precision training: take suitably scaled steps
    dr.backward(scaler.scale(loss))
    scaler.step(opt)

# Done optimizing, now let's plot the result
t = dr.linspace(Float32, 0, 1, res)
p = Array2f(dr.meshgrid(t, t))
img = Array3f(net(nn.CoopVec(p)))

# Convert 'img' with shape 3 x (N*N) into a N x N x 3 tensor
img = dr.reshape(TensorXf(img, flip_axes=True), (res, res, 3))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(ref)
ax[1].imshow(dr.clip(img, 0, 1))
fig.tight_layout()
plt.show()
