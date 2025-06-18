
from tqdm.auto import tqdm
import imageio.v3 as iio
import drjit as dr
import drjit.nn as nn
from drjit.hgrid import HashGridEncoding, PermutoEncoding
from drjit.opt import Adam, GradScaler
from drjit.auto.ad import Texture2f, TensorXf, TensorXf16, Float16, Float32, Array2f, Array3f

dr.set_flag(dr.JitFlag.KernelHistory, True)
# dr.set_log_level(dr.LogLevel.Trace)

# Load a test image and construct a texture object
ref = TensorXf(iio.imread("https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/wave-128.png") / 256)
tex = Texture2f(ref)

 # Ensure consistent results when re-running the following
dr.seed(0)

def run(config: dict):
    name = config["name"]

    n_levels = 16

    with dr.profile_range(f"run {name}"):

        result = {
            "name": name,
            "it": [],
            "loss": [],
            "mse": [],
            "d_exec": 0
        }

        extra = config.get("extra", {})
        if config["encoding"] == "hashgrid":
            encoding =  HashGridEncoding(2, n_levels, 2, **extra)
        elif config["encoding"] == "permuto":
            encoding = PermutoEncoding(2, n_levels, 2, **extra)

        encoding = encoding.alloc(Float16)

        # Establish the network structure
        net = nn.Sequential(
            nn.Cast(Float16),
            nn.Linear(-1, 3, bias=False),
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

        res = 256
        n = 10_000
        b = 100

        iterator = tqdm(range(b + n))
        for i in iterator:
            with dr.profile_range("iteration"):
                # Update network state from optimizer
                weights[:] = Float16(opt["weights"])
                encoding.data[:] = Float16(opt["data"])

                # Generate jittered positions on [0, 1]^2
                t = dr.arange(Float32, res)
                p = (Array2f(dr.meshgrid(t, t)) + dr.rand(Array2f, (2, res * res))) / res

                dr.kernel_history_clear()
                # Evaluate neural net + L2 loss
                with dr.profile_range("fwd"):
                    img = Array3f(net(nn.CoopVec(encoding(p))))
                with dr.profile_range("loss"):
                    loss = dr.squared_norm(tex.eval(p) - img)

                    dr.eval(loss)

                    if i % 100 == 0:
                        result["it"].append(i)
                        result["loss"].append(loss[0])
                    else:
                        result["loss"][-1] += loss[0]

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

        # Done optimizing, now let's plot the result
        t = dr.linspace(Float32, 0, 1, res)
        p = Array2f(dr.meshgrid(t, t))
        img = Array3f(net(nn.CoopVec(encoding(p))))

        # Convert 'img' with shape 3 x (N*N) into a N x N x 3 tensor
        img = dr.reshape(TensorXf(img, flip_axes=True), (res, res, 3))

        result["img"] = img

        result["d_exec"] /= n

        return result

configs = [
    {
        "name": "hashgrid",
        "encoding": "hashgrid",
    },
    {
        "name": "hashgrid-tcnn",
        "encoding": "hashgrid",
        "extra":{
            "torchngp_compat": True,
        }
    },
    {
        "name": "permuto",
        "encoding": "permuto",
    },
]

results = [run(config) for config in configs]
# results = [run(name) for name in ["permuto"]]
# results = [run(name) for name in ["simplifiedpermuto"]]

for result in results:
    print(f"name={result['name']}, d_exec={result['d_exec']}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2 + len(results))
for result in results:
    ax[0].plot(result["it"], result["loss"], label = result["name"])
    ax[0].legend()
    ax[0].set_yscale("log")
ax[1].set_title("Reference")
ax[1].imshow(ref)
for i in range(len(results)):
    result = results[i]
    ax[2+i].set_title(result["name"])
    ax[2+i].imshow(result["img"])
plt.show()

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].imshow(ref)
# ax[1].imshow(dr.clip(img, 0, 1))
# fig.tight_layout()
# plt.show()
