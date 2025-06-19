
from copy import deepcopy
from tqdm.auto import tqdm
import imageio.v3 as iio
import drjit as dr
import drjit.nn as nn
from drjit.hgrid import HashGridEncoding, PermutoEncoding
from drjit.opt import Adam, GradScaler
from drjit.auto.ad import (
    Texture2f,
    TensorXf,
    TensorXf16,
    Float16,
    Float32,
    Array2f,
    Array3f,
    ArrayXf,
)
import collections


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

dr.set_flag(dr.JitFlag.KernelHistory, True)
# dr.set_log_level(dr.LogLevel.Trace)

# Load a test image and construct a texture object
ref = iio.imread("data/albert.jpg") / 256
ref = ref.reshape([ref.shape[0], ref.shape[1], 1])
ref = TensorXf(ref)
tex = Texture2f(ref)

height, width, channels = ref.shape

 # Ensure consistent results when re-running the following
dr.seed(0)

def validate(result, net, encoding):
    # Done optimizing, now let's plot the result
    u = dr.linspace(Float32, 0, 1, width)
    v = dr.linspace(Float32, 0, 1, height)
    p = Array2f(dr.meshgrid(u, v))
    img = ArrayXf(net(nn.CoopVec(encoding(p))))

    img = dr.reshape(TensorXf(img, flip_axes=True), (height, width, channels))

    mse = dr.mean(dr.square(img.array - ref.array), axis = None)
    result["mse"].append(mse)

def run(config: dict):

    tmp = deepcopy(default_config)
    update(tmp, config)
    config = tmp

    name = config["name"]

    with dr.profile_range(f"run {name}"):

        result = {
            "name": name,
            "it": [],
            "loss": [],
            "mse": [],
            "d_exec": 0
        }

        encoding = config["encoding"]
        if encoding["type"] == "hashgrid":
            encoding =  HashGridEncoding(2, **encoding["config"])
        elif encoding["type"] == "permuto":
            encoding = PermutoEncoding(2, **encoding["config"])

        encoding = encoding.alloc(Float16)

        # Establish the network structure
        net = nn.Sequential(
            nn.Cast(Float16),
            nn.Linear(-1, 1, bias=False),
            nn.Exp()
        )

        # Instantiate the network for a specific backend + input size
        net = net.alloc(TensorXf16, encoding.out_features)

        # Convert to training-optimal layout
        weights, net = nn.pack(net, layout='training')

        # Optimize a single-precision copy of the parameters
        opt = Adam(lr=1e-3, params={"data": Float32(encoding.data), "weights": weights})

        # This is an adaptive mixed-precision (AMP) optimization, where a half
        # precision computation runs within a larger single-precision program.
        # Gradient scaling is required to make this numerically well-behaved.
        scaler = GradScaler()

        batch_size = config["batch_size"]
        n = config["iterations"]
        b = config["burnin"]

        iterator = tqdm(range(b + n))
        for i in iterator:
            with dr.profile_range("iteration"):
                # Update network state from optimizer
                weights[:] = Float16(opt["weights"])
                encoding.data[:] = Float16(opt["data"])

                # Generate jittered positions on [0, 1]^2
                p = dr.rand(Array2f, (2, batch_size))
                # t = dr.arange(Float32, res)
                # p = (Array2f(dr.meshgrid(t, t)) + dr.rand(Array2f, (2, res * res))) / res

                dr.kernel_history_clear()
                # Evaluate neural net + L2 loss
                with dr.profile_range("fwd"):
                    img = ArrayXf(net(nn.CoopVec(encoding(p))))
                with dr.profile_range("loss"):
                    img_ref = ArrayXf(tex.eval(p))
                    loss = dr.squared_norm(img_ref - img)

                    dr.eval(loss)

                iterator.set_postfix({"loss": loss[0]})

                # Mixed-precision training: take suitably scaled steps
                with dr.profile_range("bwd"):
                    dr.backward(scaler.scale(loss))


                history = dr.kernel_history()
                if (i > b):
                    result["d_exec"] += (
                        dr.sum([kernel["execution_time"] for kernel in history])
                    )

                with dr.profile_range("step"):
                    scaler.step(opt)

                if i % config["validation_interval"] == 0:
                    validate(result, net, encoding)

                    result["it"].append(i)
                    result["loss"].append(loss[0])

        # Done optimizing, now let's plot the result
        u = dr.linspace(Float32, 0, 1, width)
        v = dr.linspace(Float32, 0, 1, height)
        p = Array2f(dr.meshgrid(u, v))
        img = ArrayXf(net(nn.CoopVec(encoding(p))))

        # Convert 'img' with shape 3 x (N*N) into a N x N x 3 tensor
        img = dr.reshape(TensorXf(img, flip_axes=True), (height, width, channels))

        result["img"] = img

        result["d_exec"] /= n

        return result

configs = [
    {
        "name": "hashgrid",
        "encoding": {"type": "hashgrid"},
    },
    {
        "name": "hashgrid-tcnn",
        "encoding": {
            "type": "hashgrid",
            "config": {
                "torchngp_compat": True,
            },
        },
    },
    # {
    #     "name": "permuto",
    #     "encoding": "permuto",
    # },
]

default_config = {
    "batch_size": 2**16,
    "iterations": 1_000,
    "validation_interval": 100,
    "burnin": 100,
    "encoding": {
        "config": {
            "n_levels": 16,
            "n_features_per_level": 2,
        }
    },
}

results = [run(config) for config in configs]
# results = [run(name) for name in ["permuto"]]
# results = [run(name) for name in ["simplifiedpermuto"]]

for result in results:
    print(f"name={result['name']}, d_exec={result['d_exec']}")
print(f"{result=}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1 + len(results))
for result in results:
    ax[0][0].plot(result["it"], result["loss"], label = result["name"])
    ax[0][0].legend()
    ax[0][0].set_yscale("log")
ax[0][0].set_title("loss")

for result in results:
    ax[0][1].plot(result["it"], result["mse"], label = result["name"])
    ax[0][1].legend()
    ax[0][1].set_yscale("log")
ax[0][1].set_title("MSE")

ax[1][0].set_title("Reference")
ax[1][0].imshow(ref)
for i in range(len(results)):
    result = results[i]
    ax[1][1+i].set_title(result["name"])
    ax[1][1+i].imshow(result["img"])
plt.show()

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].imshow(ref)
# ax[1].imshow(dr.clip(img, 0, 1))
# fig.tight_layout()
# plt.show()
