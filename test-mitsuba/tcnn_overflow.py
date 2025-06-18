import tinycudann as tcnn
import torch

# A hashgrid with a per_level_scale of 2, base resolution of 16 and 16 levels
# will result in some levels, where the grid_resolution^D will be smaller than
# the hashmap_size for that level due to integer overflows.
# One level, where this becomes very obvious is level 12, which will have a
# grid_resolution of 2^16. This will result in a stride of 0 for the last dimension.
# Since the hashmap_size for that level is also 2^16, the indexing function
# will not use hash indexing. A larger hashmap size can also cause the same issue.
config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 1,
    "log2_hashmap_size": 16,
    "base_resolution": 16,
    "per_level_scale": 2,
    "interpolation": "Nearest", # Nearest neighbor interpolation, to investigate indexing
}

# We use a hashgrid with 3 dimension, since this results in an obvious error,
# where the last dimension of any input point is ignored for certain layers.
# To be certain about the output values we use float32 types.
hg = tcnn.Encoding(
    3,
    config,
    dtype=torch.float32,
)

# We construct the data in a way, that no two parameters have the same value
data = hg.params.data
data = torch.linspace(0, 1_000, data.shape[0], device = "cuda", dtype = torch.float32)
hg.params.data = data

# We sample the last dimension with 100 samples and interval 0.01.
x = torch.tensor([[0.1, 0.1, i * 0.01] for i in range(100) ], device = "cuda")
print("x=")
print(x)

# Evaluate the hashgrid
res = hg(x)

# One of the problematic layers is layer 12 where `hashmap_size == 2^16` and `grid_resolution == 2^16`.
res12 = res[:, 12]
print("result of layer 12=")
print(res12)

# If hash indexing was used, we would expect the features for different input
# coordinates to be different from one another. This asserts that the bug exists.
assert torch.all(res12 == res12[0])

