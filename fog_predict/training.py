from dataset import VideoDataset
from model import ResNet3D,MC3Model
from torch.utils.data import DataLoader, random_split
import torch,os
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labels_list = pd.read_csv('./output/output.csv').iloc[:,3]
dataset = VideoDataset(video_paths=os.listdir('./input_video'), labels=labels_list)

train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model=MC3Model(num_classes=4).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# 選擇ExponentialLR學習率調度器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# 訓練模型
num_epochs = 100
early_stopping_patience = 30
best_val_loss = float('inf')
early_stopping_counter = 0
train_losses, val_losses=[],[]
train_acc, val_acc=[],[]
for epoch in range(num_epochs):
    print('begin training at epoch {}'.format(epoch))
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 將數據移動到GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy= correct / total


    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy= correct / total
    val_loss /= len(test_loader)
    train_losses.append(running_loss/len(train_loader))
    val_losses.append(val_loss)
    train_acc.append(train_accuracy)
    val_acc.append(accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, accuracy: {accuracy:.4f}')
    # 指數學習率調度器
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print('Early stopping')
        break

# # 評估模型
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()
# test_loss = 0.0
# predict_list,label_list=[],[]
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()
#         predict_list.append(outputs.cpu().numpy()[0][0])
#         label_list.append(labels.cpu().numpy()[0][0])
# test_loss /= len(test_loader)
# accuracy = r2_score(y_true=label_list, y_pred=predict_list)
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

# 繪製損失和準確率曲線
def plot_metric(train_metric, val_metric, metric_name):
    plt.plot(train_metric, 'blue', label=f'Train {metric_name}')
    plt.plot(val_metric, 'red', label=f'Validation {metric_name}')
    plt.title(f'{metric_name} vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(metric_name+'trend.jpg')


plot_metric(train_losses, val_losses, 'Loss')
plot_metric(train_acc, val_acc, 'Accuracy')