import sys
sys.path.append("/Users/ctb/WorkSpace/EasyDeepRecommend")
import torch.nn as nn
import torch.nn.functional as F
from DeepRecommand.pytorch.FeatureEmbedding.criteo_feature_embedding import CriteoFeatureEmbedding


class Wide(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Wide, self).__init__()
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=output_dim)
        self.relu = nn.ReLU()
    def forward(self, X):
        return self.linear(X)


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


class WideDeep(nn.Module):
    def __init__(self, feature_map, model_config):
        super(WideDeep, self).__init__()
        self.numeric_feature_len = feature_map["numeric_feature_len"]
        self.categorical_feature_len = feature_map["categorical_feature_len"]
        self.input_dim = feature_map["sample_len"]
        if model_config["hidden_layers"][0] != self.input_dim:
            model_config["hidden_layers"].insert(0, self.input_dim)   # 因为第一个线性层的input_dim要等于样本长度
        if model_config['hidden_layers'][-1] != 1:
            model_config['hidden_layers'].append(1)
        self.hidden_layers = model_config['hidden_layers']
        self.dropout_p = model_config["dropout_p"]
        
        self.embedding_layer = CriteoFeatureEmbedding(feature_map=feature_map)
        
        self.wide = Wide(input_dim=self.numeric_feature_len)
        self.deep = Deep(hidden_layers=self.hidden_layers, dropout_p=self.dropout_p)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.embedding_layer(X)
        wide_out = self.wide(X[:,:self.numeric_feature_len])    # wide部分仅输入稠密dense特征，即数值型特征
        deep_out = self.deep(X)
        res = self.sigmoid((wide_out + deep_out) * 0.5)
        return res
    

if __name__ == "__main__":
    feature_map = {}
    features_map = {
        "I1": {"vocab_size": 3, "feature_dim": 2},
        "I2": {"vocab_size": 3, "feature_dim": 2}
    }
    feature_map["features_map"] = features_map

    X = {
        "label":[0, 1, 1],
        "I1": [1, 2, 0],
        "I2": [0, 2, 1]
    }

    model = WideDeep(feature_map=feature_map,
                     dim_in=4,
                     dim_out=1)
    
    res = model(X)
    print(f"\nres:\n {res}")
    print(f"\nres.shape: {res.shape}")









