import torch
from torch import nn

# Input data: temperature conversions from Celsius to Fahrenheit
X1 = torch.tensor([[10]], dtype=torch.float32)  # 10°C
y1 = torch.tensor([[50]], dtype=torch.float32)  # need to be a matrix even if it is one element

X2 = torch.tensor([[37.78]], dtype=torch.float32)  # 37.78°C (body temperature)
y2 = torch.tensor([[100.0]])  # 100°F

# Initialize model, loss function and optimizer
model = nn.Linear(1,1)  # Simple linear model (y = wx + b)
loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # Stochastic Gradient Descent

# y1_pred = model(X1)

# print(loss_fn(y1_pred, 
#              torch.tensor([5.7327]) ))

# for i in model.parameters():
#     print(i)
# print(model.parameters())

print('weight before training:', model.weight.item())
print('bias before training:', model.bias.item())
# Training Pass
for i in range(100000):
   # First training example (10°C -> 50°F)
   optimizer.zero_grad()  # Reset gradients
   outputs = model(X1)
   loss = loss_fn(outputs, y1)
   loss.backward()  # calculating the gradients
   optimizer.step()  # performing the optimization step

   # Second training example (37.78°C -> 100°F)
   optimizer.zero_grad()
   outputs = model(X2)
   loss = loss_fn(outputs, y2)
   loss.backward()
   optimizer.step()

   # if i%750 == 0:
   #     print(model.bias.item())
   #     print(model.weight.item())

print('weight after training:', model.weight.item())
print('bias after training:', model.bias.item())