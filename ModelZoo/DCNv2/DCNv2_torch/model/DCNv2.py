import torch
import torch.nn as nn
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
    

class CrossNetV2(nn.Module):
    """
    直接使用矩阵乘法的CrossNet
    Args:
        input_dim: 输入维度
        layer_num: CrossNet的层数
    """
    def __init__(self, input_dim, layer_num=2):
        super(CrossNetV2, self).__init__()
        self.input_dim = input_dim
        self.layer_num = layer_num
        self.cross_net = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                       for _ in range(self.layer_num))
    
    def forward(self, x_0):
        x_l = x_0
        for l in range(self.layer_num):
            x_l = x_0 * self.cross_net[l](x_l) + x_l    # linear其实就是矩阵乘法，原文公式：x_i+1 = x_0 · （W_i · x_l + b_l） + x_i
        return x_l


class CrossNetV2_with_MoE(nn.Module):
    """
    使用MoE+低秩分解的CrossNet
    Args:
        input_dim: 输入维度
        layer_num: CrossNet的层数
        num_experts: 专家数量
        low_rank: 低秩
    """
    def __init__(self, input_dim, layer_num=2, num_experts=4, low_rank=32):
        super(CrossNetV2_with_MoE, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts
        self.low_rank = low_rank
        
        # U_list: layer_num * (num_experts, input_dim, low_rank)
        self.U_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank))) for _ in range(self.layer_num)])
        # V_list: layer_num * (num_experts, input_dim, low_rank)    
        self.V_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank))) for _ in range(self.layer_num)])
        # C_list: layer_num * (num_experts, low_rank, low_rank)
        self.C_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))) for _ in range(self.layer_num)])
    
        # 初始化gates, 有多个专家，就有多少个gate，因为gate就相当于是专家的权重, 维度：num_experts * (input_dim, 1)
        self.gates = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(self.num_experts)])
        
        # 初始化bias，一层MoE相当于是CrossNet的一个layer，所以有多少层就有多少个bias, 维度：layer_num * (input_dim, 1)
        self.bias = nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(input_dim, 1))) for _ in range(self.layer_num)])
        
    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: 输入，维度：(batch_size, input_dim)
        Returns:
            outputs: 输出，维度：(batch_size, input_dim)
        """
        x_0 = inputs.unsqueeze(2)  # (bs, input_dim, 1)   
        x_l = x_0

        for l in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []

            for i in range(self.num_experts):
                # Step1: 计算gate_score，即专家权重，需要注意的是，同一个专家，无论在CrossNet的哪一层，其gate的初始化值都是一样的；
                #        但同一个专家，在不同层，其gate_score是不同的，因为x_l是不断变化的
                gating_score_of_experts.append(self.gates[i](x_l.squeeze(2)))   # (bs, 1)

                # Step2: 进行低秩特征交叉，原文公式：
                # E_i(x_l) = x_0 * {
                #       U_l_i · tanh(C_l_i · tanh(V_l_i^T · x_l)) + b_l
                # } 
                # 其中，*表示Hadamard-product，即逐元素相乘； ·表示矩阵乘法； tanh表示tanh激活函数,即：使用非线性变换来提炼特征
                # 
                v_x = torch.matmul(self.V_list[l][i].t(), x_l)  # (bs, input_dim, low_rank)^T * (bs, input_dim, 1) = (bs, low_rank, 1)
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[l][i], v_x)      # (bs, low_rank, low_rank) * (bs, low_rank, 1) = (bs, low_rank, 1)
                v_x = torch.tanh(v_x)
                uv_x = torch.matmul(self.U_list[l][i], v_x)     # (bs, input_dim, low_rank) * (bs, low_rank, 1) = (bs, input_dim, 1)
                dot_ = uv_x + self.bias[l]                      # (bs, input_dim, 1) + (bs, input_dim, 1) = (bs, input_dim, 1)
                dot_ = x_0 * dot_                               # (bs, input_dim, 1) * (bs, input_dim, 1) = (bs, input_dim, 1), 即Hadamard乘积，逐元素相乘

                output_of_experts.append(dot_.squeeze(2))

            # Step3: 对cross_net的每一层，综合专家结果，得到最终的输出x_l
            output_of_experts = torch.stack(output_of_experts, 2)                           # num_experts * (bs, input_dim) -> (bs, input_dim, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)               # num_experts * (bs, 1) -> (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))   # (bs, input_dim, num_experts) * (bs, num_experts, 1) = (bs, input_dim, 1)
            x_l = moe_out + x_l                                                             # (bs, input_dim, 1)

        return x_l.squeeze(2)   # (bs, input_dim)
    

class DCNv2(nn.Module):
    def __init__(self, feature_map, model_config):
        super(DCNv2, self).__init__()

        # 获取CrossNet相关配置
        self.use_low_rank_and_moe = model_config['use_low_rank_and_moe']
        self.model_structure = model_config['model_structure']
        self.num_cross_layers = model_config['num_cross_layers']
        self.num_experts = model_config['num_experts']
        self.low_rank = model_config['low_rank']

        # 获取Deep相关配置
        self.input_dim = feature_map["sample_len"]
        if model_config["hidden_layers"][0] != self.input_dim:
            model_config["hidden_layers"].insert(0, self.input_dim)
        self.hidden_layers = model_config['hidden_layers']
        self.dropout_p = model_config['dropout_p']
        
        # 初始化embedding层
        self.embedding_layer = CriteoFeatureEmbedding(feature_map=feature_map)

        # 初始化CrossNet模型
        if self.use_low_rank_and_moe:
            self.cross_net = CrossNetV2_with_MoE(self.input_dim, self.num_cross_layers, self.num_experts, self.low_rank)
        else:
            self.cross_net = CrossNetV2(self.input_dim, self.num_cross_layers)

        # 初始化Deep模型
        self.deep = Deep(self.hidden_layers, self.dropout_p)

        # 计算最后输出层的dim
        if self.model_structure == "crossnet_only":
            self.final_dim = self.input_dim
        elif self.model_structure == "stacked":             # 串行，先crossNet，再Deep
            self.final_dim = self.hidden_layers[-1]
        elif self.model_structure == "parallel":            # 并行, Deep和CrossNet并行
            self.final_dim = self.hidden_layers[-1] + self.input_dim
            
        # 初始化输出层
        self.out = nn.Linear(self.final_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        """
        Args:
            X: 输入数据, shape=(batch_size, input_dim)
        Returns:
            y_pred: 预测值, shape=(batch_size, 1)
        """
        X = self.embedding_layer(X)

        if self.model_structure == "crossnet_only":
            x_l = self.cross_net(X)
            res = self.out(x_l)
        elif self.model_structure == "stacked":
            x_l = self.cross_net(X)
            res = self.out(x_l) 
        elif self.model_structure == "parallel":
            x_l = self.cross_net(X)
            deep_out = self.deep(X)
            res = self.out(torch.cat([x_l, deep_out], dim=1))

        res = self.sigmoid(res)
        return res      
            



        