# Initial values
X1 = 37.78  # Feature value
y = 100     # Target value
w1 = 0.5    # Weight parameter
b = 0       # Bias parameter

# What ∂L/∂w1 and ∂L/∂b mean:
# ------------------------------------
# Step 1: Calculate current prediction
prediction = X1 * w1 + b
print(f"Current prediction: {prediction}")  # 18.89

# Step 2: Calculate current error
error = prediction - y
print(f"Current error: {error}")  # -81.11

# Step 3: Calculate current loss (MSE)
loss = error**2
print(f"Current loss: {loss}")  # 6578.83

# Step 4: Calculate derivative for w1
dL_dw1 = 2 * X1 * error
print(f"∂L/∂w1 = {dL_dw1}")  # -6124.15

# Step 5: Calculate derivative for b
dL_db = 2 * error
print(f"∂L/∂b = {dL_db}")  # -162.23

# Example 1: Effect of increasing w1 by 0.001
# -------------------------------------------
new_w1 = w1 + 0.001
new_prediction_w1 = X1 * new_w1 + b
new_error_w1 = new_prediction_w1 - y
new_loss_w1 = new_error_w1**2

print(f"\nAfter increasing w1 by 0.001:")
print(f"New loss: {new_loss_w1}")  # 6571.08
print(f"Actual decrease: {loss - new_loss_w1}")  # 7.75
print(f"Expected decrease (from derivative): {-dL_dw1 * 0.001}")  # 6.12

# Example 2: Effect of increasing b by 0.001
# ------------------------------------------
new_b = b + 0.001
new_prediction_b = X1 * w1 + new_b
new_error_b = new_prediction_b - y
new_loss_b = new_error_b**2

print(f"\nAfter increasing b by 0.001:")
print(f"New loss: {new_loss_b}")  # 6578.67
print(f"Actual decrease: {loss - new_loss_b}")  # 0.16
print(f"Expected decrease (from derivative): {-dL_db * 0.001}")  # 0.16

# Meaning of the derivatives:
# --------------------------
# 1. ∂L/∂w1 = -6124.15: This large negative value means increasing w1
#    significantly decreases the loss. The weight parameter needs to
#    increase substantially to better fit the data.
#
# 2. ∂L/∂b = -162.23: This negative value means increasing b
#    decreases the loss, but the effect is smaller than changing w1.
#
# The derivatives guide gradient descent by showing:
# - Direction to adjust parameters (opposite sign of derivative)
# - Relative importance of each parameter (magnitude of derivative)
# - How sensitive the loss is to small changes in parameters


# The gradient always points in the direction of the steepest increase of the function. 
# Since we want to minimize the loss function (not maximize it), 
# we need to go in the opposite direction of the gradient.
# This is the fundamental principle behind gradient descent. 
# The gradient tells us which way is "uphill" for the loss function, 
# and we want to go "downhill," so we move in the opposite direction.



