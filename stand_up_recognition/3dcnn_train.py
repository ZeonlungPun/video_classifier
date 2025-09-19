from torchvision.models.video import mc3_18,r3d_18,r2plus1d_18
from video_classifier_dataset import StandUpDataset
import os,torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from collections import Counter,defaultdict
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix, ConfusionMatrixDisplay
from simple_3D_CNN import Small3DCNN



def inference(model, val_dl, class_names,device):
    model.eval()
    correct_prediction = 0
    total_prediction = 0
    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in tqdm(val_dl, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Normalize
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 統計整體正確率
            correct_prediction += (preds == labels).sum().item()
            total_prediction += preds.size(0)

            # 統計分類正確率
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    # 整體正確率
    acc = correct_prediction / total_prediction
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'\n✅ Overall Accuracy: {acc:.2f}, Macro-F1: {macro_f1:.2f}')
    print(f'Total items: {total_prediction}')
    unique_classes = np.unique(np.concatenate((all_labels, all_preds)))
    class_names = [class_names[i] for i in unique_classes]
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f'Confusion Matrix (Acc: {acc:.4f}, F1: {macro_f1:.4f})')
    plt.tight_layout()
    plt.savefig('cm.png')

    # 每個類別的正確率
    print('\n📊 Per-Class Accuracy:')
    for i, class_name in enumerate(class_names):
        total = class_total[i]
        correct = class_correct[i]
        acc = correct / total if total > 0 else 0.0
        print(f'  {class_name}: {acc:.2f} ({correct}/{total})')
    return macro_f1
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, val_dl, class_names,save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights=torch.tensor([1,1,1.25]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001,
        steps_per_epoch=len(train_dl),
        epochs=num_epochs,
        anneal_strategy='linear'
    )
    best_macro_f1 = 0.0  # ← 初始化最佳驗證準確率

    # 🔁 如果有已保存的模型，讀取模型參數並載入
    if os.path.exists(save_path):
        print(f"📥 Loading existing model from '{save_path}'...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        # 一次驗證，更新 best_macro_f1：
        if val_dl is not None:
            best_macro_f1 = inference(model, val_dl, class_names,device)
            print(f"🔍 Loaded model macro-F1: {best_macro_f1:.4f}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # 🔄 使用 tqdm 顯示每個 epoch 的進度
        loop = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for inputs, labels in loop:

            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            # Normalize
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # 🟦 更新 tqdm 顯示內容
            loop.set_postfix(loss=loss.item(), acc=correct_prediction / total_prediction)

        # 每個 epoch 結束後列印結果
        epoch_loss = running_loss / len(train_dl)
        epoch_acc = correct_prediction / total_prediction
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

        # 每幾個 epoch 呼叫一次驗證函數
        if val_dl is not None and epoch%5==0:
            macro_f1=inference(model, val_dl, class_names,device)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), save_path)
                print(f'🎯 Best model updated and saved (Macro-F1: {macro_f1:.4f})')

class TemporalAvgPoolWrapper(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # 先拿到原本的輸出維度
        in_features = backbone.fc.in_features
        # 去掉原本的 fc
        self.backbone.fc = nn.Identity()
        # 自己定義一個新的分類層
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        feat = self.backbone.stem(x)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)
        feat = self.backbone.avgpool(feat)  # (B, C, T, 1, 1)

        # ✨ 在時間維度 (T) 上取平均
        feat = feat.mean(dim=2)             # (B, C, 1, 1)

        feat = torch.flatten(feat, 1)       # (B, C)
        out = self.fc(feat)                 # (B, num_classes)
        return out

if __name__ == '__main__':

    train_img_path_list=[]
    test_img_path_list=[]
    train_label_list,test_label_list=[],[]
    train_path='/home/zonekey/project/3d_train_test/train'
    test_path ='/home/zonekey/project/3d_train_test/test'
    class_path=os.listdir(train_path)
    for cls in class_path:
        cls_img_list_train=os.listdir(os.path.join(train_path,cls))
        for img_path in cls_img_list_train:
            train_img_path_list.append(os.path.join(train_path,cls,img_path))
            train_label_list.append(cls)

        cls_img_list_test = os.listdir(os.path.join(test_path, cls))
        for img_path in cls_img_list_test:
            test_img_path_list.append(os.path.join(test_path, cls, img_path))
            test_label_list.append(cls)

    #train_label_list=list(map(lambda x:x.replace('2','0'),train_label_list))
    #test_label_list = list(map(lambda x: x.replace('2', '0'), test_label_list))
    label_counter_train = Counter(train_label_list)
    label_counter_test =Counter(test_label_list)
    sample_weights_train = []
    for label in train_label_list:
        if label == "stand_up":
            sample_weights_train.append(1.5 / label_counter_train[label])  # 提高 stand_up 權重
        else:
            sample_weights_train.append(1.0 / label_counter_train[label])

    #sample_weights_train = [1.0 / label_counter_train[label] for label in train_label_list]
    sample_weights_test = [1.0  for label in test_label_list]
    sampler_train = WeightedRandomSampler(
        weights=sample_weights_train,
        num_samples=len(sample_weights_train),
        replacement=True
    )
    sampler_test = WeightedRandomSampler(
        weights=sample_weights_test,
        num_samples=len(sample_weights_test),
        replacement=True
    )

    trainDataset=StandUpDataset(dir_list=train_img_path_list,new_size=(224,112))
    train_loader = DataLoader(dataset=trainDataset, batch_size=10,sampler=sampler_train)
    testDataset = StandUpDataset(dir_list=test_img_path_list, new_size=(224,112),mode='test')
    test_loader = DataLoader(dataset=testDataset, batch_size=10,sampler=sampler_test)

    # build the video classifier model
    #model = mc3_18(pretrained=True)
    model =r3d_18(pretrained=True)

    num_classes = 3
    #model = Small3DCNN(num_classes=num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model.classifier[1] = torch.nn.Conv3d(in_channels=1024, out_channels=4, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    # 重新構建 classifier 部分（仍保留 Dropout）
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.2),
    #     nn.Conv3d(in_channels=1024, out_channels=num_classes, kernel_size=1)
    # )

    #base_model = mc3_18(pretrained=True)
    #model = TemporalAvgPoolWrapper(base_model, num_classes)
    class_names=['always_sit','movement','stand_up']
    # set the training parameters
    num_epochs = 100
    training(model, train_loader, num_epochs,test_loader, class_names, save_path='r3d18_action_3class_frozen.pth')