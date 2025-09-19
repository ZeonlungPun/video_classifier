import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os,time
import numpy as np
from torchvision.models.video import mc3_18,r3d_18
import torch.nn as nn

device = torch.device( "cpu")
class_names =  ['always_sit',  'movement', 'stand_up']

def load_model(model_name):
    num_classes = len(class_names)
    if model_name=='mc3':
        model = mc3_18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load('./mc3_action_3class.pth', map_location=device))

    else:
        model = r3d_18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load('./r3d18_action_3class_frozen.pth', map_location=device))

    model.to(device)
    model.eval()
    return model

def uniform_sample(img_list, num_frames):
    if len(img_list) < num_frames:
        # 不足就補最後一幀
        img_list.extend([img_list[-1]] * (num_frames - len(img_list)))
    indices = np.linspace(0, len(img_list)-1, num_frames).astype(int)
    return [img_list[i] for i in indices]


# 輸入前處理（要和訓練時一致）
transform = transforms.Compose([
    transforms.Resize((224, 112)),  # 例如 (H, W)
    transforms.ToTensor(),

])

def load_video_frames(img_series_path, frame_num=10):
    all_img_name = os.listdir(img_series_path)
    all_img_name = [f for f in all_img_name if f.endswith('.jpg')]
    img_list=[]
    for img_name in all_img_name:
        full_img_path = os.path.join(img_series_path, img_name)
        img_pil = Image.open(full_img_path).convert("RGB")
        img_list.append(img_pil)
    transformed_img_list=[]
    for img in img_list:
        transformed_img_list.append(transform(img))
    transformed_img_list = uniform_sample(transformed_img_list, frame_num)
    ## (T, C, H, W) --> ( C,T, H, W)
    clip_tensor = torch.stack(transformed_img_list).permute((1, 0, 2, 3))
    # ( C,T, H, W) --> (1, C,T, H, W)
    return clip_tensor.unsqueeze(0)


def inference(video_path, model, class_names):
    inputs = load_video_frames(video_path)
    inputs = inputs.to(device)
    t1=time.time()
    with torch.no_grad():
        outputs = model(inputs)         # [1, num_classes]
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    t2=time.time()
    print('time:',t2-t1)
    return class_names[pred], probs.cpu().numpy()[0]
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate_all(category_list,sample_path,model):
    sample_path_list= os.listdir(sample_path)
    correct_count= 0
    total_count =0
    for path in sample_path_list:
        all_path= os.path.join("/home/zonekey/project/3d_train_test/test/movement/",path)
        label_name, probs = inference(all_path, model, category_list)
        pred_label = int(category_list.index(label_name))
        true_label = int(category_list.index(sample_path.split('/')[-1]))
        if true_label == pred_label:
            correct_count+=1
        total_count+=1

    acc= correct_count/total_count
    print("accuracy:",acc)





if __name__ == '__main__':
    model =load_model('r3d18')
    video_path = "/home/zonekey/project/3d_train_test/test/stand_up/t188.07-p0"
    label, probs = inference(video_path, model, class_names)

    print("Prediction:", label)
    print("Probabilities:", dict(zip(class_names, probs)))

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    sample_path='/home/zonekey/project/3d_train_test/test/movement'
    #evaluate_all(class_names,sample_path, model)