import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import zarr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Pytorch’s LSTM expects all of its inputs to be 3D tensors. 
# The semantics of the axes of these tensors is important. 
# The first axis is the sequence itself, the second indexes instances in the mini-batch, 
# and the third indexes elements of the input. We haven’t discussed mini-batching, 
# so let’s just ignore that and assume we will always have just 1 dimension on the second axis.

# 1. Data Preparation
def read_cf_pair(dt, n):
    start, end = dt.index_range[n]
    timestamps = dt.timestamp[start:end]
    t = (timestamps-timestamps[0]) # read time and convert it to second

    lead_centroid = dt.lead_centroid[start: end]
    follow_centroid = dt.follow_centroid[start: end]
    follow_velocity = dt.follow_velocity[start: end]
    lead_velocity = dt.lead_velocity[start: end]
    follow_acceleration = dt.follow_acceleration[start: end]
    lead_acceleration = dt.lead_acceleration[start: end]
    # size_lead = dt.lead_size[n] # this is for HV
    size_lead = 4.8
    size_follow = dt.follow_size[n]

    return lead_centroid, follow_centroid, follow_velocity, follow_acceleration, lead_acceleration, lead_velocity, t, size_lead, size_follow

# For easier data manipulation, we convert the data into a pandas dataframe
# x_lead, x_follow, v_follow, a_follow, a_lead, v_lead, t, size_lead, size_follow = read_cf_pair(data, 120)

zarr_data = zarr.open('../data/valAV.zarr', mode='r')
data = pd.DataFrame(data=dict(zip(['x_lead', 'x_follow', 'v_follow', 'a_follow', 'a_lead', 'v_lead', 't', 'size_lead', 'size_follow'], read_cf_pair(zarr_data, 3054))))

for i in range (0, 50):
    data = data.append(pd.DataFrame(data=dict(zip(['x_lead', 'x_follow', 'v_follow', 'a_follow', 'a_lead', 'v_lead', 't', 'size_lead', 'size_follow'], read_cf_pair(zarr_data, i)))))

# Calculate spacing and relative speed
data['spacing'] = data['x_lead'] - data['x_follow'] - data['size_lead']
data['relative_speed'] = data['v_lead'] - data['v_follow']

# Select features and target
features = data[['spacing', 'relative_speed', 'v_follow', 'v_lead', 'a_follow' ,'a_lead', 'x_lead']]
target = data['x_follow']

# Normalize data
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values)
 
# 2. Model Definition
class CarFollowingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CarFollowingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :]) 
        return out

input_dim = X_train.shape[1]
hidden_dim = 32
num_layers = 2
output_dim = 1

model = CarFollowingLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)

# 3. Training
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_function(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 4. Saving
torch.save(model.state_dict(), 'car_following_model.pth')

# 5. Evaluation
model.eval()
test_outputs = model(X_test_tensor)
test_loss = loss_function(test_outputs, y_test_tensor)
print(f'Test Loss: {test_loss.item()}')

y_pred = test_outputs.detach().numpy()
y_true = y_test_tensor.numpy()
mse = mean_squared_error(y_true, y_pred)
print(f'MSE on test set: {mse}')


# Plotting the results
import matplotlib.pyplot as plt

x_lead = leader_data['x_lead'].values
x_follow_actual = follower_data['x_follow'].values

# LSTM model prediction
# Prepare input features for LSTM prediction (based on the features you've used for training)
input_features = follower_data[['feature1', 'feature2', ...]].values 
# Normalize and reshape the input features
input_features_scaled = scaler.transform(input_features) # Assuming 'scaler' was used to normalize the training data
input_features_tensor = torch.FloatTensor(input_features_scaled).unsqueeze(0) # Add batch dimension
# Predict using the trained LSTM model
model.eval() # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(input_features_tensor)
# Convert predictions to NumPy array
x_follow_pred = predictions.numpy().squeeze()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_lead, label='Leader Position', color='blue')
plt.plot(x_follow_actual, label='Follower Actual Position', color='green')
plt.plot(x_follow_pred, label='Follower Predicted Position', color='red', linestyle='dashed')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.title('Leader and Follower Positions Over Time')
plt.show()