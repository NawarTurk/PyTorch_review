import torch
from torch import nn

# Temp examples in C - freezing, room temp, boiling
X = torch.tensor([
    [0.0],
    [25.0],  
    [100.0]  
])

# Linear model - nothing fancy
# 1,1 means 1 input feature, 1 output feature
# bias=True is default, so not specifying it
model = nn.Linear(1, 1)  # bias=True is implicit

# Let's see what random params PyTorch gave us
print("BEFORE:")
print(model)
# bias is a vector, weight is a matrix (even for single values!)
print(f"bias: {model.bias.item():.4f}")  # .item() gets the scalar value
print(f"weight: {model.weight.item():.4f}")  # here weight is a 1x1 matrix

# This will be garbage with random weights
print("\nRandom predictions (nonsense):")
print(model(X))

# Actually we know the C->F formula is F = 1.8C + 32
# So let's just set those values directly!
print("\nFIXING IT...")
model.weight = nn.Parameter(torch.tensor([[1.8]]))  # multiply by 1.8 - this is a MATRIX (2D)
model.bias = nn.Parameter(torch.tensor([32.0]))     # add 32 - this is a VECTOR (1D)

# Check our handiwork
print(f"New bias: {model.bias.item()}")  # should be 32
print(f"New weight: {model.weight.item()}")  # should be 1.8

# Now we should get proper F temps
temps_f = model(X)
print("\nPredictions in Fahrenheit:")
print(temps_f)

# Quick sanity check
print("\nMake sure these look right:")
print(f"0°C → {temps_f[0].item()}°F")      # should be 32°F
print(f"25°C → {temps_f[1].item()}°F")     # should be 77°F
print(f"100°C → {temps_f[2].item()}°F")    # should be 212°F