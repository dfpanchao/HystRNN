import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv

# Set random seed for reproducibility
torch.manual_seed(42)

# Load array from the CSV file in another file
loaded_data_B_Major_FORC = np.loadtxt('Data/B_unsat_Major2FORC.csv', delimiter=',')
loaded_data_H_Major_FORC = np.loadtxt('Data/H_unsat_Major2FORC.csv', delimiter=',')
loaded_data_B_amp1 = np.loadtxt('Data/B_unsat_amp1.csv', delimiter=',')
loaded_data_H_amp1 = np.loadtxt('Data/H_unsat_amp1.csv', delimiter=',')
loaded_data_B_amp2 = np.loadtxt('Data/B_unsat_amp2.csv', delimiter=',')
loaded_data_H_amp2 = np.loadtxt('Data/H_unsat_amp2.csv', delimiter=',')

B_Major_FORC = loaded_data_B_Major_FORC.reshape(-1,1)
H_Major_FORC = loaded_data_H_Major_FORC.reshape(-1,1)
B_amp1 = loaded_data_B_amp1.reshape(-1,1)
H_amp1 = loaded_data_H_amp1.reshape(-1,1)
B_amp2 = loaded_data_B_amp2.reshape(-1,1)
H_amp2 = loaded_data_H_amp2.reshape(-1,1)

def bias_function(x):
    return 0.1 * np.sin(2 * np.pi * x) + 0.001 * np.random.normal(size=len(x))

holder = B_Major_FORC
x = np.linspace(np.min(holder), np.max(holder), holder.shape[0])
y_bias = bias_function(x).reshape(-1,1)
B_Major_FORC = holder + y_bias

holder = B_amp1
x = np.linspace(np.min(holder), np.max(holder), holder.shape[0])
y_bias = bias_function(x).reshape(-1,1)
B_amp1 = holder + y_bias

holder = B_amp2
x = np.linspace(np.min(holder), np.max(holder), holder.shape[0])
y_bias = bias_function(x).reshape(-1,1)
B_amp2 = holder + y_bias

# plt.plot(B_amp2_unbiassed)
# plt.plot(B_amp2)
# plt.show()
# quit()

B_major = B_Major_FORC[:595]
H_major = H_Major_FORC[:595]

B1 = B_Major_FORC[596:795]
H1 = H_Major_FORC[596:795]

B2 = B_Major_FORC[995:]
H2 = H_Major_FORC[995:]
#
# plt.plot(H_major, B_major)
# plt.plot(H1, B1)
# plt.plot(H2, B2)
# plt.show()
# quit()

B_amp1 = loaded_data_B_amp1.reshape(-1,1)
H_amp1 = loaded_data_H_amp1.reshape(-1,1)
B_amp2 = loaded_data_B_amp2.reshape(-1,1)
H_amp2 = loaded_data_H_amp2.reshape(-1,1)

##########################################################
##################### Scaling the data ###################
##########################################################
def min_max_scaling(X):
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = (X - X_min) / (X_max - X_min) * 2 - 1
    return X_scaled, X_min, X_max

def min_max_scaling_inverse(X_scaled, X_min, X_max):
    X_original = (X_scaled + 1) / 2 * (X_max - X_min) + X_min
    return X_original

# Min-Max Scaling
B_Major_FORC_scaled, B_min, B_max = min_max_scaling(loaded_data_B_Major_FORC)
H_Major_FORC_scaled, H_min, H_max = min_max_scaling(loaded_data_H_Major_FORC)

B_Major_FORC_scaled = B_Major_FORC_scaled.reshape(-1,1)
H_Major_FORC_scaled = H_Major_FORC_scaled.reshape(-1,1)

B_major_scaled_train = B_Major_FORC_scaled[:595]
H_major_scaled_train = H_Major_FORC_scaled[:595]

B1_scaled_test = B_Major_FORC_scaled[596:795]
H1_scaled_test = H_Major_FORC_scaled[596:795]

B2_scaled_test = B_Major_FORC_scaled[995:]
H2_scaled_test = H_Major_FORC_scaled[995:]

B_amp1_scaled = (loaded_data_B_amp1 - B_min) / (B_max - B_min) * 2 - 1
H_amp1_scaled = (loaded_data_H_amp1 - H_min) / (H_max - H_min) * 2 - 1

B_amp1_scaled_test = B_amp1_scaled.reshape(-1,1)
H_amp1_scaled_test = H_amp1_scaled.reshape(-1,1)

B_amp2_scaled = (loaded_data_B_amp2 - B_min) / (B_max - B_min) * 2 - 1
H_amp2_scaled = (loaded_data_H_amp2 - H_min) / (H_max - H_min) * 2 - 1

B_amp2_scaled_test = B_amp2_scaled.reshape(-1,1)
H_amp2_scaled_test = H_amp2_scaled.reshape(-1,1)

###################################################
########### Train and test set preparation ########
###################################################

x_train = np.concatenate([B_major_scaled_train[:-1], H_major_scaled_train[:-1]], axis=1)
y_train = B_major_scaled_train[1:]

x_test_1 = np.concatenate([B1_scaled_test[0].reshape(-1,1), H1_scaled_test[0].reshape(-1,1)], axis=1)
y_test_1 = B1_scaled_test[1:]

x_test_2 = np.concatenate([B2_scaled_test[0].reshape(-1,1), H2_scaled_test[0].reshape(-1,1)], axis=1)
y_test_2 = B2_scaled_test[1:]

x_test_amp1 = np.concatenate([B_amp1_scaled_test[0].reshape(-1,1), H_amp1_scaled_test[0].reshape(-1,1)], axis=1)
y_test_amp1 = B_amp1_scaled_test[1:]

x_test_amp2 = np.concatenate([B_amp2_scaled_test[0].reshape(-1,1), H_amp2_scaled_test[0].reshape(-1,1)], axis=1)
y_test_amp2 = B_amp2_scaled_test[1:]

###################################################
####################### NN ########################
###################################################

# Size
input_size = 2
hidden_size = 32
output_size = 1
sequence_length = len(B_major_scaled_train[:-1])
batch_size = 1
num_epochs = 10000

# Convert data to tensors
input_tensor = torch.tensor(x_train).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(y_train).view(batch_size, sequence_length, output_size).float()

#Convert test data
test_input_tensor_1 = torch.tensor(x_test_1).view(batch_size, -1, input_size).float()
test_target_tensor_1 = torch.tensor(y_test_1).view(batch_size, -1, output_size).float()

test_input_tensor_2 = torch.tensor(x_test_2).view(batch_size, -1, input_size).float()
test_target_tensor_2 = torch.tensor(y_test_2).view(batch_size, -1, output_size).float()

test_input_tensor_amp1 = torch.tensor(x_test_amp1).view(batch_size, -1, input_size).float()
test_target_tensor_amp1 = torch.tensor(y_test_amp1).view(batch_size, -1, output_size).float()

test_input_tensor_amp2 = torch.tensor(x_test_amp2).view(batch_size, -1, input_size).float()
test_target_tensor_amp2 = torch.tensor(y_test_amp2).view(batch_size, -1, output_size).float()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

# Set random seed for reproducibility
torch.manual_seed(42)

model = RNN(input_size, hidden_size, output_size)

# Step 4: Implement the training loop
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Set initial hidden state
    hidden = torch.zeros(1, batch_size, hidden_size)

    # Forward pass
    output, hidden = model(input_tensor, hidden)
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.8f}')


with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    #cell_pred = torch.zeros(1, batch_size, hidden_size)
    prediction_1, _ = model(test_input_tensor_1, hidden_pred)
    prediction_tensor_1 = torch.zeros(len(B1_scaled_test[:-1]), 1)
    prediction_tensor_1[0] = prediction_1
    for i in range(len(B1_scaled_test[:-1])-1):
        hidden_pred = torch.zeros(1, batch_size, hidden_size)
        #cell_pred = torch.zeros(1, batch_size, hidden_size)
        x_test_B_1 = prediction_1.reshape(-1, 1)
        x_test_H_1 = H1_scaled_test[i+1].reshape(-1, 1)
        x_test_1 = np.concatenate([x_test_B_1, x_test_H_1], axis=1)
        test_input_tensor_1 = torch.tensor(x_test_1).view(batch_size, -1, input_size).float()
        prediction_1, _ = model(test_input_tensor_1, hidden_pred)
        prediction_tensor_1[i+1] = prediction_1

# Flatten prediction tensor
prediction_1 = prediction_tensor_1.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_1 = np.array(prediction_1)
y_test_1 = min_max_scaling_inverse(y_test_1, B_min, B_max)

with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    #cell_pred = torch.zeros(1, batch_size, hidden_size)
    prediction_2, _ = model(test_input_tensor_2, hidden_pred)
    prediction_tensor_2 = torch.zeros(len(B2_scaled_test[:-1]), 1)
    prediction_tensor_2[0] = prediction_2
    for i in range(len(B2_scaled_test[:-1])-1):
        hidden_pred = torch.zeros(1, batch_size, hidden_size)
        #cell_pred = torch.zeros(1, batch_size, hidden_size)
        x_test_B_2 = prediction_2.reshape(-1, 1)
        x_test_H_2 = H2_scaled_test[i+1].reshape(-1, 1)
        x_test_2 = np.concatenate([x_test_B_2, x_test_H_2], axis=1)
        test_input_tensor_2 = torch.tensor(x_test_2).view(batch_size, -1, input_size).float()
        prediction_2, _ = model(test_input_tensor_2, hidden_pred)
        prediction_tensor_2[i+1] = prediction_2
#
# Flatten prediction tensor
prediction_2 = prediction_tensor_2.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_2 = np.array(prediction_2)
y_test_2 = min_max_scaling_inverse(y_test_2, B_min, B_max)
#
with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    #cell_pred = torch.zeros(1, batch_size, hidden_size)
    prediction_amp1, _ = model(test_input_tensor_amp1, hidden_pred)
    prediction_tensor_amp1 = torch.zeros(len(B_amp1_scaled_test[:-1]), 1)
    prediction_tensor_amp1[0] = prediction_amp1
    for i in range(len(B_amp1_scaled_test[:-1])-1):
        hidden_pred = torch.zeros(1, batch_size, hidden_size)
        #cell_pred = torch.zeros(1, batch_size, hidden_size)
        x_test_B_amp1 = prediction_amp1.reshape(-1, 1)
        x_test_H_amp1 = H_amp1_scaled_test[i+1].reshape(-1, 1)
        x_test_1_amp1 = np.concatenate([x_test_B_amp1, x_test_H_amp1], axis=1)
        test_input_tensor_amp1 = torch.tensor(x_test_1_amp1).view(batch_size, -1, input_size).float()
        prediction_amp1, _ = model(test_input_tensor_amp1, hidden_pred)
        prediction_tensor_amp1[i+1] = prediction_amp1
#
# Flatten prediction tensor
prediction_amp1 = prediction_tensor_amp1.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_amp1 = np.array(prediction_amp1)
#
with torch.no_grad():
    hidden_pred = torch.zeros(1, batch_size, hidden_size)
    #cell_pred = torch.zeros(1, batch_size, hidden_size)
    prediction_amp2, _ = model(test_input_tensor_amp2, hidden_pred)
    prediction_tensor_amp2 = torch.zeros(len(B_amp2_scaled_test[:-1]), 1)
    prediction_tensor_amp2[0] = prediction_amp2
    for i in range(len(B_amp2_scaled_test[:-1])-1):
        hidden_pred = torch.zeros(1, batch_size, hidden_size)
        #cell_pred = torch.zeros(1, batch_size, hidden_size)
        x_test_B_amp2 = prediction_amp2.reshape(-1, 1)
        x_test_H_amp2 = H_amp2_scaled_test[i+1].reshape(-1, 1)
        x_test_1_amp2 = np.concatenate([x_test_B_amp2, x_test_H_amp2], axis=1)
        test_input_tensor_amp2 = torch.tensor(x_test_1_amp2).view(batch_size, -1, input_size).float()
        prediction_amp2, _ = model(test_input_tensor_amp2, hidden_pred)
        prediction_tensor_amp2[i+1] = prediction_amp2
#
# Flatten prediction tensor
prediction_amp2 = prediction_tensor_amp2.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_amp2 = np.array(prediction_amp2)

test_output_data_1 = min_max_scaling_inverse(test_output_data_1, B_min, B_max)
test_output_data_2 = min_max_scaling_inverse(test_output_data_2, B_min, B_max)
test_output_data_amp1 = min_max_scaling_inverse(test_output_data_amp1, B_min, B_max)
test_output_data_amp2 = min_max_scaling_inverse(test_output_data_amp2, B_min, B_max)
y_test_1 = min_max_scaling_inverse(y_test_1, B_min, B_max)
y_test_2 = min_max_scaling_inverse(y_test_2, B_min, B_max)
y_test_amp1 = min_max_scaling_inverse(y_test_amp1, B_min, B_max)
y_test_amp2 = min_max_scaling_inverse(y_test_amp2, B_min, B_max)

print("#######################################")

# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((test_output_data_1.reshape(-1,1) - y_test_1)**2) / np.mean(y_test_1**2)
print("Relative Error Test FORC 1 RNN : ", relative_error_test)
relative_error_test = np.mean((test_output_data_2.reshape(-1,1) - y_test_2)**2) / np.mean(y_test_2**2)
print("Relative Error Test FORC 2 RNN: ", relative_error_test)
relative_error_test = np.mean((test_output_data_amp1.reshape(-1,1) - y_test_amp1)**2) / np.mean(y_test_amp1**2)
print("Relative Error Test amp 1 RNN: ", relative_error_test)
relative_error_test = np.mean((test_output_data_amp2.reshape(-1,1) - y_test_amp2)**2) / np.mean(y_test_amp2**2)
print("Relative Error Test amp 2 RNN: ", relative_error_test)

print("#######################################")

R_abs = np.max(np.abs(test_output_data_1.reshape(-1,1) - y_test_1))
print("Max error FORC 1 RNN : ", R_abs)
R_abs = np.max(np.abs(test_output_data_2.reshape(-1,1) - y_test_2))
print("Max error FORC 2 RNN : ", R_abs)
R_abs = np.max(np.abs(test_output_data_amp1.reshape(-1,1) - y_test_amp1))
print("Max error amp 1 RNN : ", R_abs)
R_abs = np.max(np.abs(test_output_data_amp2.reshape(-1,1) - y_test_amp2))
print("Max error amp 2 RNN : ", R_abs)

print("#######################################")

mean_y = np.mean(y_test_1)
numerator = np.var(mean_y - test_output_data_1.reshape(-1,1))
denominator = np.var(y_test_1)
evs = 1 - numerator / denominator
print("Explained Variance Score FORC 1 RNN:", evs)

mean_y = np.mean(y_test_2)
numerator = np.var(mean_y - test_output_data_2.reshape(-1,1))
denominator = np.var(y_test_2)
evs = 1 - numerator / denominator
print("Explained Variance Score FORC 2 RNN:", evs)

mean_y = np.mean(y_test_amp1)
numerator = np.var(mean_y - test_output_data_amp1.reshape(-1,1))
denominator = np.var(y_test_amp1)
evs = 1 - numerator / denominator
print("Explained Variance Score amp 1 RNN:", evs)

mean_y = np.mean(y_test_amp2)
numerator = np.var(mean_y - test_output_data_amp2.reshape(-1,1))
denominator = np.var(y_test_amp2)
evs = 1 - numerator / denominator
print("Explained Variance Score amp 2 RNN:", evs)

print("#######################################")

MAE = np.mean(np.abs(test_output_data_1.reshape(-1,1) - y_test_1))
print("MAE FORC 1 RNN: ", MAE)
MAE = np.mean(np.abs(test_output_data_2.reshape(-1,1) - y_test_2))
print("MAE FORC 2 RNN: ", MAE)
MAE = np.mean(np.abs(test_output_data_amp1.reshape(-1,1) - y_test_amp1))
print("MAE amp 1 RNN: ", MAE)
MAE = np.mean(np.abs(test_output_data_amp2.reshape(-1,1) - y_test_amp2))
print("MAE amp 2 RNN: ", MAE)

#####################################################
################# Fig. 3a ###########################
#####################################################

# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(5, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

# Plot the data with red and blue lines, one with dotted and one with solid style
ax.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3, label='Major loop')
ax.plot(H1, B1, color='red', linestyle='solid', linewidth=3, label='FORC')
ax.plot(H1[1:], test_output_data_1, color='black', linestyle='dashdot', linewidth=3, label='pred')

#ax.plot(H2, B2, color='red', linestyle='solid', linewidth=7, label='FORC')


# Set the axis labels with bold font weight
ax.set_xlabel(r"H[A/m]", fontsize=28, color='black')
ax.set_ylabel(r"B[T]", fontsize=28, color='black')

# Set the number of ticks for x-axis and y-axis to 3
ax.set_xticks([-125, 0, 125])
ax.set_yticks([-1.25, 0, 1.25])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

# Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.savefig('Figures/dc_1_RNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()

#####################################################
################# Fig. 3d ###########################
#####################################################

# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(5, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

# Plot the data with red and blue lines, one with dotted and one with solid style
ax.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3, label='Major loop')
ax.plot(H2, B2, color='red', linestyle='solid', linewidth=3, label='FORC')
ax.plot(H2[1:], test_output_data_2, color='black', linestyle='dashdot', linewidth=3, label='pred')


# Set the axis labels with bold font weight
ax.set_xlabel(r"H[A/m]", fontsize=28, color='black')
ax.set_ylabel(r"B[T]", fontsize=28, color='black')

# Set the number of ticks for x-axis and y-axis to 3
ax.set_xticks([-125, 0, 125])
ax.set_yticks([-1.25, 0, 1.25])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

#Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.savefig('Figures/dc_2_RNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()

#####################################################
################# Fig. 3g ###########################
#####################################################

# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(5, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

# Plot the data with red and blue lines, one with dotted and one with solid style
ax.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3, label='Major loop')
ax.plot(H_amp1, B_amp1, color='red', linestyle='solid', linewidth=3, label='FORC')
ax.plot(H_amp1[1:], test_output_data_amp1, color='black', linestyle='dashdot', linewidth=3, label='pred')


# Set the axis labels with bold font weight
ax.set_xlabel(r"H[A/m]", fontsize=28, color='black')
ax.set_ylabel(r"B[T]", fontsize=28, color='black')

# Set the number of ticks for x-axis and y-axis to 3
ax.set_xticks([-125, 0, 125])
ax.set_yticks([-1.25, 0, 1.25])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

#Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.savefig('Figures/dc_3_RNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()

#####################################################
################# Fig. 3j ###########################
#####################################################


# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(5, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

# Plot the data with red and blue lines, one with dotted and one with solid style
ax.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3, label='Major loop')
ax.plot(H_amp2, B_amp2, color='red', linestyle='solid', linewidth=3, label='FORC')
ax.plot(H_amp2[1:], test_output_data_amp2, color='black', linestyle='dashdot', linewidth=3, label='pred')


# Set the axis labels with bold font weight
ax.set_xlabel(r"H[A/m]", fontsize=28, color='black')
ax.set_ylabel(r"B[T]", fontsize=28, color='black')

# Set the number of ticks for x-axis and y-axis to 3
ax.set_xticks([-125, 0, 125])
ax.set_yticks([-1.25, 0, 1.25])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

#Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.savefig('Figures/dc_4_RNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()
