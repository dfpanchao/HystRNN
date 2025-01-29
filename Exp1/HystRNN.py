import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import explained_variance_score

# Set random seed for reproducibility
torch.manual_seed(42)

# Load array from the CSV file in another file
loaded_data_B_Major_FORC = np.loadtxt('Data/B_Major2FORC.csv', delimiter=',')
loaded_data_H_Major_FORC = np.loadtxt('Data/H_Major2FORC.csv', delimiter=',')
loaded_data_B_amp1 = np.loadtxt('Data/B_amp1.csv', delimiter=',')
loaded_data_H_amp1 = np.loadtxt('Data/H_amp1.csv', delimiter=',')
loaded_data_B_amp2 = np.loadtxt('Data/B_amp2.csv', delimiter=',')
loaded_data_H_amp2 = np.loadtxt('Data/H_amp2.csv', delimiter=',')

B_Major_FORC = loaded_data_B_Major_FORC.reshape(-1,1)
H_Major_FORC = loaded_data_H_Major_FORC.reshape(-1,1)
B_amp1 = loaded_data_B_amp1.reshape(-1,1)
H_amp1 = loaded_data_H_amp1.reshape(-1,1)
B_amp2 = loaded_data_B_amp2.reshape(-1,1)
H_amp2 = loaded_data_H_amp2.reshape(-1,1)

B_major = B_Major_FORC[:595]
H_major = H_Major_FORC[:595]

B1 = B_Major_FORC[596:795]
H1 = H_Major_FORC[596:795]

B2 = B_Major_FORC[995:]
H2 = H_Major_FORC[995:]

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

class HystRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt):
        super(HystRNNCell, self).__init__()
        self.dt = dt
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)
        self.i2p = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self, x, hy, hz, hp):
        hp = torch.tanh(self.i2p(torch.cat((torch.abs(x)**2, torch.abs(hz)**2, torch.abs(hy)**2), 1)))
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1))) + hp)
        hy = hy + self.dt * hz

        return hy, hz, hp

class HystRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt):
        super(HystRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = HystRNNCell(n_inp, n_hid, dt)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):
        hy = torch.zeros(x.size(1), self.n_hid)
        hz = torch.zeros(x.size(1), self.n_hid)
        hp = torch.zeros(x.size(1), self.n_hid)

        for t in range(x.size(0)):
            hy, hz, hp = self.cell(x[t], hy, hz, hp)
        output = self.readout(hy)

        return output

model = HystRNN(input_size, hidden_size, output_size, dt=0.05)

# Step 4: Implement the training loop
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    # Forward pass
    output = model(input_tensor)
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}")

with torch.no_grad():
    prediction_1 = model(test_input_tensor_1)
    prediction_tensor_1 = torch.zeros(len(B1_scaled_test[:-1]), 1)
    prediction_tensor_1[0] = prediction_1
    for i in range(len(B1_scaled_test[:-1])-1):
        x_test_B_1 = prediction_1
        x_test_H_1 = H1_scaled_test[i+1].reshape(-1, 1)
        x_test_1 = np.concatenate([x_test_B_1, x_test_H_1], axis=1)
        test_input_tensor_1 = torch.tensor(x_test_1).view(batch_size, -1, input_size).float()
        prediction_1 = model(test_input_tensor_1)
        prediction_tensor_1[i+1] = prediction_1

# Flatten prediction tensor
prediction_1 = prediction_tensor_1.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_1 = np.array(prediction_1)
y_test_1 = min_max_scaling_inverse(y_test_1, B_min, B_max)

with torch.no_grad():
    prediction_2 = model(test_input_tensor_2)
    prediction_tensor_2 = torch.zeros(len(B2_scaled_test[:-1]), 1)
    prediction_tensor_2[0] = prediction_2
    for i in range(len(B2_scaled_test[:-1])-1):
        x_test_B_2 = prediction_2
        x_test_H_2 = H2_scaled_test[i+1].reshape(-1, 1)
        x_test_2 = np.concatenate([x_test_B_2, x_test_H_2], axis=1)
        test_input_tensor_2 = torch.tensor(x_test_2).view(batch_size, -1, input_size).float()
        prediction_2 = model(test_input_tensor_2)
        prediction_tensor_2[i+1] = prediction_2

# Flatten prediction tensor
prediction_2 = prediction_tensor_2.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_2 = np.array(prediction_2)
y_test_2 = min_max_scaling_inverse(y_test_2, B_min, B_max)

with torch.no_grad():
    prediction_amp1 = model(test_input_tensor_amp1)
    prediction_tensor_amp1 = torch.zeros(len(B_amp1_scaled_test[:-1]), 1)
    prediction_tensor_amp1[0] = prediction_amp1
    for i in range(len(B_amp1_scaled_test[:-1])-1):
        x_test_B_amp1 = prediction_amp1
        x_test_H_amp1 = H_amp1_scaled_test[i+1].reshape(-1, 1)
        x_test_1_amp1 = np.concatenate([x_test_B_amp1, x_test_H_amp1], axis=1)
        test_input_tensor_amp1 = torch.tensor(x_test_1_amp1).view(batch_size, -1, input_size).float()
        prediction_amp1 = model(test_input_tensor_amp1)
        prediction_tensor_amp1[i+1] = prediction_amp1

# Flatten prediction tensor
prediction_amp1 = prediction_tensor_amp1.view(-1).numpy()
# Reshape the test_output_data and y_test to numpy arrays for plotting
test_output_data_amp1 = np.array(prediction_amp1)

with torch.no_grad():
    prediction_amp2 = model(test_input_tensor_amp2)
    prediction_tensor_amp2 = torch.zeros(len(B_amp2_scaled_test[:-1]), 1)
    prediction_tensor_amp2[0] = prediction_amp2
    for i in range(len(B_amp2_scaled_test[:-1])-1):
        x_test_B_amp2 = prediction_amp2
        x_test_H_amp2 = H_amp2_scaled_test[i+1].reshape(-1, 1)
        x_test_1_amp2 = np.concatenate([x_test_B_amp2, x_test_H_amp2], axis=1)
        test_input_tensor_amp2 = torch.tensor(x_test_1_amp2).view(batch_size, -1, input_size).float()
        prediction_amp2 = model(test_input_tensor_amp2)
        prediction_tensor_amp2[i+1] = prediction_amp2

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
print("Relative Error Test FORC 1 HystRNN : ", relative_error_test)
relative_error_test = np.mean((test_output_data_2.reshape(-1,1) - y_test_2)**2) / np.mean(y_test_2**2)
print("Relative Error Test FORC 2 HystRNN: ", relative_error_test)
relative_error_test = np.mean((test_output_data_amp1.reshape(-1,1) - y_test_amp1)**2) / np.mean(y_test_amp1**2)
print("Relative Error Test amp 1 HystRNN: ", relative_error_test)
relative_error_test = np.mean((test_output_data_amp2.reshape(-1,1) - y_test_amp2)**2) / np.mean(y_test_amp2**2)
print("Relative Error Test amp 2 HystRNN: ", relative_error_test)

print("#######################################")

R_abs = np.max(np.abs(test_output_data_1.reshape(-1,1) - y_test_1))
print("Max error FORC 1 HystRNN : ", R_abs)
R_abs = np.max(np.abs(test_output_data_2.reshape(-1,1) - y_test_2))
print("Max error FORC 2 HystRNN : ", R_abs)
R_abs = np.max(np.abs(test_output_data_amp1.reshape(-1,1) - y_test_amp1))
print("Max error amp 1 HystRNN : ", R_abs)
R_abs = np.max(np.abs(test_output_data_amp2.reshape(-1,1) - y_test_amp2))
print("Max error amp 2 HystRNN : ", R_abs)

print("#######################################")

evs = explained_variance_score(y_test_1, test_output_data_1.reshape(-1,1))
print("Explained Variance Score FORC 1 HystRNN:", evs)

evs = explained_variance_score(y_test_2, test_output_data_2.reshape(-1,1))
print("Explained Variance Score FORC 2 HystRNN:", evs)

evs = explained_variance_score(y_test_amp1, test_output_data_amp1.reshape(-1,1))
print("Explained Variance Score amp 1 HystRNN:", evs)

evs = explained_variance_score(y_test_amp2, test_output_data_amp2.reshape(-1,1))
print("Explained Variance Score amp 2 HystRNN:", evs)

print("#######################################")

MAE = np.mean(np.abs(test_output_data_1.reshape(-1,1) - y_test_1))
print("MAE FORC 1 HystRNN: ", MAE)
MAE = np.mean(np.abs(test_output_data_2.reshape(-1,1) - y_test_2))
print("MAE FORC 2 HystRNN: ", MAE)
MAE = np.mean(np.abs(test_output_data_amp1.reshape(-1,1) - y_test_amp1))
print("MAE amp 1 HystRNN: ", MAE)
MAE = np.mean(np.abs(test_output_data_amp2.reshape(-1,1) - y_test_amp2))
print("MAE amp 2 HystRNN: ", MAE)


#####################################################
################# Fig. 2a ###########################
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
ax.set_xticks([-6000, 0, 6000])
ax.set_yticks([-1.7, 0, 1.7])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

# Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Create an inset axis for zoomed-in region
axins = inset_axes(ax, width="30%", height="60%", loc='upper left')

# Plot the zoomed-in region
axins.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3)
axins.plot(H1, B1, color='red', linestyle='solid', linewidth=3)
axins.plot(H1[1:], test_output_data_1, color='black', linestyle='dashdot', linewidth=3)
# Set the limits for the zoomed-in region
axins.set_xlim(-25, 175)
axins.set_ylim(0, 1.5)

# Set labels and ticks for the zoomed-in region
axins.set_xticks([-25, 75, 175])
axins.set_yticks([0, 0.75])

# Set tick labels fontweight to bold and increase font size for the inset axis
axins.tick_params(axis='both', which='major', labelsize=20, width=1, length=4)

# Set the spines linewidth to bold for the inset axis
axins.spines['top'].set_linewidth(1)
axins.spines['right'].set_linewidth(1)
axins.spines['bottom'].set_linewidth(1)
axins.spines['left'].set_linewidth(1)

plt.savefig('Figures/a2_HystRNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()

#####################################################
################# Fig. 2d ###########################
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
ax.set_xticks([-6000, 0, 6000])
ax.set_yticks([-1.7, 0, 1.7])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

#Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Create an inset axis for zoomed-in region
axins = inset_axes(ax, width="30%", height="60%", loc='upper left')

# Plot the zoomed-in region
axins.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3)
axins.plot(H2, B2, color='red', linestyle='solid', linewidth=3)
axins.plot(H2[1:], test_output_data_2, color='black', linestyle='dashdot', linewidth=3)
# Set the limits for the zoomed-in region
axins.set_xlim(-50, 125)
axins.set_ylim(-0.75, 0.75)

# Set labels and ticks for the zoomed-in region
axins.set_xticks([-50, 35, 125])
axins.set_yticks([-0.75, 0.0])

# Set tick labels fontweight to bold and increase font size for the inset axis
axins.tick_params(axis='both', which='major', labelsize=20, width=1, length=4)

# Set the spines linewidth to bold for the inset axis
axins.spines['top'].set_linewidth(1)
axins.spines['right'].set_linewidth(1)
axins.spines['bottom'].set_linewidth(1)
axins.spines['left'].set_linewidth(1)

plt.savefig('Figures/d2_HystRNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()

#####################################################
################# Fig. 2g ###########################
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
ax.set_xticks([-6000, 0, 6000])
ax.set_yticks([-1.7, 0, 1.7])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

#Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Create an inset axis for zoomed-in region
axins = inset_axes(ax, width="30%", height="60%", loc='upper left')

# Plot the zoomed-in region
axins.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3)
axins.plot(H_amp1, B_amp1, color='red', linestyle='solid', linewidth=3)
axins.plot(H_amp1[1:], test_output_data_amp1, color='black', linestyle='dashdot', linewidth=3)
# Set the limits for the zoomed-in region
axins.set_xlim(-175, 175)
axins.set_ylim(-1.5, 1.5)

# Set labels and ticks for the zoomed-in region
axins.set_xticks([-175, 0, 175])
axins.set_yticks([-1.5, 0.0])

# Set tick labels fontweight to bold and increase font size for the inset axis
axins.tick_params(axis='both', which='major', labelsize=20, width=1, length=4)

# Set the spines linewidth to bold for the inset axis
axins.spines['top'].set_linewidth(1)
axins.spines['right'].set_linewidth(1)
axins.spines['bottom'].set_linewidth(1)
axins.spines['left'].set_linewidth(1)

plt.savefig('Figures/g2_HystRNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()

#####################################################
################# Fig. 2j ###########################
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
ax.set_xticks([-6000, 0, 6000])
ax.set_yticks([-1.7, 0, 1.7])

# Set tick labels fontweight to bold and increase font size
ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=10)

#Set the spines linewidth to bold
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Create an inset axis for zoomed-in region
axins = inset_axes(ax, width="30%", height="60%", loc='upper left')

# Plot the zoomed-in region
axins.plot(H_major, B_major, color='blue', linestyle='solid', linewidth=3)
axins.plot(H_amp2, B_amp2, color='red', linestyle='solid', linewidth=3)
axins.plot(H_amp2[1:], test_output_data_amp2, color='black', linestyle='dashdot', linewidth=3)

# Set the limits for the zoomed-in region
axins.set_xlim(-100, 125)
axins.set_ylim(-1.3, 1.3)

# Set labels and ticks for the zoomed-in region
axins.set_xticks([-100, 20, 125])
axins.set_yticks([-1.3, 0.0])

# Set tick labels fontweight to bold and increase font size for the inset axis
axins.tick_params(axis='both', which='major', labelsize=20, width=1, length=4)

# Set the spines linewidth to bold for the inset axis
axins.spines['top'].set_linewidth(1)
axins.spines['right'].set_linewidth(1)
axins.spines['bottom'].set_linewidth(1)
axins.spines['left'].set_linewidth(1)

plt.savefig('Figures/j2_HystRNN.jpeg', dpi=500, bbox_inches="tight")
plt.show()
