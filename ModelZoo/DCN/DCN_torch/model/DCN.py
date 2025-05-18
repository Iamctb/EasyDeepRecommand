import torch
from torch import nn
from DeepRecommand.pytorch.FeatureEmbedding.criteo_feature_embedding import CriteoFeatureEmbedding


class Deep(nn.Module):
    def __init__(self, hidden_layers, dropout_p=0.0):
        """
        Deep: 深度网络
        Args:
            hidden_layers: deep网络的隐层维度, 如[128, 64, 32]
            dropout_p: dropout_p值. Defaults to 0.0.
        """
        super(Deep, self).__init__()
        self.dnn = nn.ModuleList()
        for layer in list(zip(hidden_layers[:-1], hidden_layers[1:])):
            linear = nn.Linear(in_features=layer[0], out_features=layer[1])
            self.dnn.append(linear)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, X):
        """
        Args:
            X: 输入数据, shape=(batch_size, input_dim)
        Returns:
            res: deep网络的输出, shape=(batch_size, hidden_layers[-1])
        """
        for linear in self.dnn:
            X = linear(X)
        res = self.dropout(X)
        return res

    
class CrossInteraction(nn.Module):
    def __init__(self, input_dim):
        """
        CrossInteraction: 交叉网络的单层
        Args:
            input_dim: 输入维度
        """
        super(CrossInteraction, self).__init__()
        self.w = nn.Linear(input_dim, 1, bias=False)     # 权重矩阵, 生成(1, input_dim)的参数矩阵 W
        self.b = nn.Parameter(torch.rand(input_dim))
    
    def forward(self, X_i, X_0):
        """
        Args:
            X_i: 本层的输入, shape=(batch_size, input_dim)
            X_0: 第0层的输入, shape=(batch_size, input_dim)
        Returns:
            out: 本层的输出, shape=(batch_size, input_dim)
        """
        out = self.w(X_i) * X_0 + self.b   # self.w(X_i) * X_0 = torch.matmul(X_i, W^T) * X_0 = torch.matmul((batch_size, input_dim), (input_dim, 1 )) * (batch_size, input_dim) = (batch_size, input_dim), 其中matmul表示矩阵乘法， *表示对应逐元素相乘
        return out


class CrossNet_v1(nn.Module):
    def __init__(self, input_dim, num_layers):
        """
        CrossNet_v1: 交叉网络
        Args:
            input_dim: 输入维度
            num_layers: cors网络的层数
        """
        super(CrossNet_v1, self).__init__()
        self.num_layers = num_layers
        self.corss_layers = nn.ModuleList(
            CrossInteraction(input_dim) for _ in range(num_layers)
        )
    
    def forward(self, X_0):
        """
        Args:
            X_0: 网络输入, shape=(batch_size, input_dim)
        Returns:
            X_i: 网络输出, shape=(batch_size, input_dim)
        """
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.corss_layers[i](X_i, X_0)  # 注意：这个地方不要写成X_i += 这种形式，因为这种形式表示：变量被原地（in-place）修改了，这导致无法正确地回溯计算梯度
        return X_i

class DCN(nn.Module):
    def __init__(self, feature_map, model_config):
        """
        DCN_v1: 即常说的DCN(Deep Cross Network)
        Args:
            feature_map: 特征map
            model_config: 模型配置
        """
        super(DCN, self).__init__()
        self.input_dim = feature_map["sample_len"]
        if model_config["hidden_layers"][0] != self.input_dim:
            model_config["hidden_layers"].insert(0, self.input_dim)   # 因为第一个线性层的input_dim要等于样本长度
        if model_config['hidden_layers'][-1] != 1:
            model_config['hidden_layers'].append(1)
        self.hidden_layers = model_config['hidden_layers']            # 深度网络的层数 
        self.num_cross_layers = model_config['num_cross_layers']
        self.dropout_p = model_config['dropout_p']

        self.embedding_layer = CriteoFeatureEmbedding(feature_map=feature_map)

        self.cross = CrossNet_v1(self.input_dim, self.num_cross_layers)
        self.deep = Deep(self.hidden_layers, self.dropout_p)
        self.fc = nn.Linear(self.hidden_layers[-1] + self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        """
        Args:
            X: 输入数据, shape=(batch_size, input_dim)
        Returns:
            y_pred: 预测值, shape=(batch_size, 1)
        """
        X = self.embedding_layer(X)
        cross_out = self.cross(X)
        deep_out = self.deep(X)
        concat_out = torch.cat([cross_out, deep_out], dim=-1)
        y_pred = self.fc(concat_out)
        y_pred = self.sigmoid(y_pred)
        return y_pred


if __name__ == "__main__":
    X = torch.randn(4, 128)
    # model = DCN_v1(128, [128, 64, 32])
    # y_pred = model(X)
    # print(y_pred.shape)