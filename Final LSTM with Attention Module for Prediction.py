import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from openvino.runtime import Core
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
file_path = 'C:/Users/Murad Khan/ene_shr/Home_energy_data.csv'
data = pd.read_csv(file_path)

# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['DateTime']))

# Prepare the dataset for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 12
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

# Create TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Increased batch size

# Define the LSTM model with attention in PyTorch
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention_weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.attention_bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.attention_weight)
        nn.init.zeros_(self.attention_bias)

    def attention(self, lstm_output):
        e = torch.tanh(torch.matmul(lstm_output, self.attention_weight) + self.attention_bias)
        a = torch.softmax(e, dim=1)
        context_vector = torch.sum(a * lstm_output, dim=1)
        return context_vector

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        context_vector = self.attention(lstm_output)
        out = self.fc(context_vector)
        return out

# Hyperparameters
input_dim = 17
hidden_dim = 32
output_dim = 17
num_layers = 2
dropout = 0.2
num_epochs = 50
learning_rate = 0.002

# Initialize model, loss function, and optimizer
model = LSTMAttentionModel(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Train the model
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    train_loss = running_loss / len(dataloader)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        val_loss = criterion(val_outputs, torch.tensor(y_test, dtype=torch.float32).to(device)).item()
        val_losses.append(val_loss)

        # Calculate training accuracy
        train_predictions = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        train_accuracy = 1 - mean_absolute_error(y_train.flatten(), train_predictions.flatten())
        train_accuracies.append(train_accuracy)

        # Calculate validation accuracy
        val_predictions = val_outputs.cpu().numpy()
        val_accuracy = 1 - mean_absolute_error(y_test.flatten(), val_predictions.flatten())
        val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Export the trained PyTorch model to ONNX
onnx_model_path = 'lstm_attention_model.onnx'
dummy_input = torch.randn(1, SEQ_LENGTH, input_dim).to(device)
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)

# Initialize OpenVINO Inference Engine
ie = Core()

# Check available devices
available_devices = ie.available_devices
print("Available devices:", available_devices)

# Ensure GPU is listed as an available device
assert "GPU" in available_devices, "GPU device not found in OpenVINO."

model_xml = "C:/Users/Murad Khan/ene_shr/output/lstm_attention_model.xml"
model_bin = "C:/Users/Murad Khan/ene_shr/output/lstm_attention_model.bin"
net = ie.read_model(model=model_xml)

# Compile the model for the GPU device
compiled_model = ie.compile_model(model=net, device_name="GPU")

# Prepare test data for prediction
X_test_tensor = torch.tensor(X_test[:24], dtype=torch.float32).cpu().numpy()

# Warm-up the GPU
for _ in range(10):
    compiled_model([X_test_tensor])

# Perform inference using OpenVINO and measure time
start_time = time.time()
res = compiled_model([X_test_tensor])[compiled_model.output(0)]
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time on GPU: {inference_time:.4f} seconds")

# Inverse transform the predictions
predictions = scaler.inverse_transform(res)

# Rescale the actual values for comparison
actual_values = scaler.inverse_transform(y_test[:24])

# Function to plot the training and validation loss
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

plot_loss(train_losses, val_losses)

# Function to plot the training and validation accuracy
def plot_accuracy(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()

plot_accuracy(train_accuracies, val_accuracies)
