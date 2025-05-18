import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim                           
current_dir = os.path.dirname(__file__)                                 # 获取当前脚本所在的目录
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))   # 定位项目根目录
sys.path.append(project_root)                                           # 添加项目根目录到 sys.path
from DeepRecommand.pytorch.dataloader.Criteo_Dataloader import CriteoDataloader
from model.DCNv2 import DCNv2
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random
import yaml

def set_seed(seed):
    """
    设置随机种子以确保实验结果的可复现性。
    参数:
    seed (int): 随机种子的值。
    """
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python的随机种子
    random.seed(seed)
    # 设置cudnn的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 禁止hash随机化
    os.environ['PYTHONHASHSEED'] = str(seed)


def trian_and_valid(data_config, feature_map, model_config, model_save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nStep2:获取数据 ...")
    train_dataloader = CriteoDataloader(data_path=data_config['train_data'], 
                                        features_name=feature_map['numeric_feature'] + feature_map['categorical_feature'],
                                        label_name=feature_map['label'],
                                        batch_size=model_config['batch_size'],
                                        shuffle=False,
                                        num_workers=0) 
    valid_dataloader = CriteoDataloader(data_path=data_config['valid_data'], 
                                        features_name=feature_map['numeric_feature'] + feature_map['categorical_feature'],
                                        label_name=feature_map['label'],
                                        batch_size=model_config['batch_size'],
                                        shuffle=False,
                                        num_workers=0)
    

    model = DCNv2(feature_map=feature_map, model_config=model_config).to(device)

    if model_config['loss'] == 'bce':
        criterion = nn.BCELoss()
    if model_config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)   # 控制学习率下降
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    num_epochs = model_config['epoch']

    print("\nStep3: 训练中")
    final_val_auc, final_val_loss = 0, 0  # 用于保存最后的AUC和loss
    for epoch in range(num_epochs):
        model.train
        train_labels, train_preds = [], []
        
        with tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training") as pbar:
            for batch_data in pbar:
                features = {k: v.to(device) for k, v in batch_data.items() if k != 'label'}
                labels = batch_data['label'].to(device)
                # 向前传播
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs.float(), labels.float())

                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 保存所预测和标签,用于后续的AUC计算
                train_labels.extend(labels.cpu().detach().numpy())
                train_preds.extend(outputs.cpu().detach().numpy())

                # 更新进度条
                pbar.set_postfix(loss=loss.item())

                # 清理缓存
                torch.cuda.empty_cache()

        # 计算训练集AUC
        train_auc = roc_auc_score(train_labels, train_preds)
        
        # 验证过程
        model.eval()
        val_labels, val_preds = [], []
        val_loss = 0.0

        with torch.no_grad():
            with tqdm(valid_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}] - validing") as pbar:
                for batch_data in pbar:
                    features = {k: v.to(device) for k, v in batch_data.items() if k != 'label'}
                    labels = batch_data['label'].to(device)
                    outputs = model(features).squeeze()
                    loss = criterion(outputs.float(), labels.float())
                    val_loss += loss.item()

                    # 保存验证集的预测和标签用于AUC计算
                    val_labels.extend(labels.cpu().detach().numpy())
                    val_preds.extend(outputs.cpu().detach().numpy())

                    # 更新进度条描述
                    pbar.set_postfix(loss=loss.item())
        
        scheduler.step(val_loss)

        val_auc = roc_auc_score(val_labels, val_preds)
        val_loss /= len(valid_dataloader)

        if epoch == num_epochs-1:
            final_val_auc = val_auc
            final_val_loss = val_loss

        # 打印当前epoch的训练和验证结果
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # 获取当前日期，并在 model_path 下创建日期目录
    current_date = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(model_save_dir, current_date)
    os.makedirs(date_dir, exist_ok=True)

    # 保存模型
    current_time = datetime.now().strftime("%H-%M-%S")
    model_filename = f"{model_save_dir}/{current_date}/{current_time}_AUC_{final_val_auc:.5f}_Loss_{final_val_loss:.5f}_Epoch_{num_epochs}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"模型已保存至: {model_filename}")


if __name__ == '__main__':
    set_seed(2024)                                                              # 固定随机种子，用于代码复现
    
    print("Step1: 获取配置各项配置 ...")
     # 获取yaml配置文件
    data_config_path = os.path.join(current_dir, "config/data_config.yaml")     # 通过相对路径获取数据配置文件
    model_config_path = os.path.join(current_dir, "config/model_config.yaml")   # 通过相对路径获取模型配置文件
    with open(data_config_path, 'r', encoding='utf-8') as file:   
        data_config = yaml.safe_load(file)['Criteo_dataset']                    # 获取Criteo_dataset下的配置
    with open(model_config_path, 'r', encoding='utf-8') as file:
        model_config = yaml.safe_load(file)['DCNv2']                              # 获取DCN下的配置

    # 获取配置中的路径，和项目路径合并后重新赋值，防止因为路径导致程序出错
    data_config['feature_map'] = project_root + data_config['feature_map']
    data_config['train_data'] = project_root + data_config['train_data']
    data_config['valid_data'] = project_root + data_config['valid_data']
    data_config['test_data'] = project_root + data_config['test_data']
    model_config['model_save_dir'] = project_root + model_config['model_save_dir']

    print("data_config: \n", json.dumps(data_config, indent=4))
    print("model_config: \n", json.dumps(model_config, indent=4))

    feature_map_path = data_config['feature_map']
    with open(feature_map_path, 'r') as file:
        feature_map = json.load(file)
    print("feature_map: \n", json.dumps(feature_map, indent=4))

    trian_and_valid(
        data_config=data_config,
        feature_map=feature_map,
        model_config=model_config,
        model_save_dir=model_config["model_save_dir"]
    )

    




