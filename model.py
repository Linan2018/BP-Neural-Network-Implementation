import pickle
import random
from collections import defaultdict

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, scale
from tqdm import tqdm

random.seed(1)
np.random.seed(1)

NODE_TYPES = ["value_activated", "value_not_activated", "value_softmax", "loss"]
EDGE_TYPES = ["weight", "activation", "softmax", "loss"]


class Node:
    """节点类型（保存计算前后的值）"""

    def __init__(self, layer, index, type_, value=None):
        self.layer = layer
        self.index = index

        assert type_ in NODE_TYPES
        self.type = type_

        self.value = value

    def update_value(self):
        pass

    def set_value(self, value):
        self.value = value


class Edge:
    """边类型（保存计算过程）"""

    def __init__(self, layer, index, type_, head_node, tail_node):
        self.layer = layer
        self.index = index

        assert type_ in EDGE_TYPES
        self.type = type_

        self.head_node = head_node
        self.tail_node = tail_node


class WeightEdge(Edge):
    """权重边"""

    def __init__(self, layer, index, head_node, tail_node, weight=0):
        super(WeightEdge, self).__init__(layer, index, "weight", head_node, tail_node)
        self.weight = weight
        self.v = 0

    def diff_local(self):
        """关于输入的偏导数"""
        return self.weight

    def update_weight(self, delta, alpha=0.01, gamma=0):
        """更新权重"""
        self.v = gamma * self.v + alpha * delta
        self.weight -= self.v

    def compute(self, x):
        """计算，用于前向传播"""
        return self.weight * x


class ActivationEdge(Edge):
    def __init__(self, layer, index, head_node, tail_node):
        super(ActivationEdge, self).__init__(layer, index, "activation", head_node, tail_node)

    def diff_local(self, x):
        """激活函数tanh偏导数"""
        return 1 / (np.cosh(x) ** 2)

    def compute(self, x):
        """计算，用于前向传播"""
        return np.tanh(x)


class SoftmaxEdge(Edge):
    """softmax边（暂时没有用到）"""

    def __init__(self, layer, index, head_node, tail_node):
        super(SoftmaxEdge, self).__init__(layer, index, "softmax", head_node, tail_node)

    def diff_local(self, x, all_):
        """softmax的偏导数"""
        i = self.head_node
        j = self.tail_node
        return all_[i] * (int(i == j) - all_[j])

    def compute(self, x, all_):
        """计算，用于前向传播"""
        sum_exp = np.sum([np.exp(i) for i in all_])
        return np.exp(x) / sum_exp


class LossEdge(Edge):
    """loss边（暂时没有用到）"""

    def __init__(self, layer, index, head_node, tail_node):
        super(LossEdge, self).__init__(layer, index, "loss", head_node, tail_node)

    def diff_local(self, x, all_):
        """交叉熵损失函数关于某个前向节点x的偏导数"""
        i = self.head_node
        j = self.tail_node
        return all_[i] * (int(i == j) - all_[j])

    def compute(self, x, all_):
        """计算，用于前向传播"""
        sum_exp = np.sum([np.exp(i) for i in all_])
        return np.exp(x) / sum_exp


class SoftmaxLossEdge(Edge):
    """softmax和loss函数边"""

    def __init__(self, layer, index, head_node, tail_node):
        super(SoftmaxLossEdge, self).__init__(layer, index, "loss", head_node, tail_node)

    def diff_local(self, all_x, all_y: list, i):
        """交叉熵损失函数和softmax复合后关于某个前向节点x的偏导数"""
        n = len(all_x)
        tmp = [np.exp(i) for i in all_x]
        sum_exp = np.sum(tmp)
        p = [t / sum_exp for t in tmp]
        return p[i] - all_y[i]

    def compute(self, all_x, all_y: list):
        """计算loss，用于前向传播"""
        n = len(all_x)
        tmp = [np.exp(i) for i in all_x]
        sum_exp = np.sum(tmp)
        p = [t / sum_exp for t in tmp]
        loss = -np.sum([np.log(p[i]) * all_y[i] for i in range(n)])
        return loss

    def compute2(self, all_x):
        """计算输出概率，用于做预测"""
        n = len(all_x)
        tmp = [np.exp(i) for i in all_x]
        sum_exp = np.sum(tmp)
        p = [t / sum_exp for t in tmp]
        return p


class FullyConnectedLayer:
    """全连接层"""

    def __init__(self, no, n_in: int, n_out: int, last):
        self.no = no  # 层标号
        self.gradient = defaultdict(dict)  # 层内参数梯度
        if last:
            self.x = {i: Node(self.no, i, "value_not_activated") for i in range(n_in)}
        else:
            self.x = last

        # 加权后的输出，即WX
        self.o = {i: Node(self.no, i, "value_not_activated") for i in range(n_out)}
        # 权重边字典
        self.w = {}
        # 激活函数边字典
        self.a = {}
        # 输出，并即将作为下一层的输入
        self.out_nodes = {i: Node(self.no + 1, i, "value_activated") for i in range(n_out)}
        # 处理w
        for i in range(n_in):
            for j in range(n_out):
                self.w[(i, j)] = WeightEdge(self.no, (i, j),
                                            head_node=i,
                                            tail_node=j,
                                            weight=np.random.uniform(-1, 1))
        # 处理a
        for j in range(n_out):
            self.a[j] = ActivationEdge(self.no, j,
                                       head_node=j,
                                       tail_node=j)

    def forward(self, y=None):
        """前向传播，并更新层内节点的值"""
        for index, node in self.o.items():
            weight_edges = [node_w for i, node_w in self.w.items() if i[1] == index]
            node.set_value(
                np.sum([weight_edge.compute(self.x[weight_edge.head_node].value) for weight_edge in weight_edges]))
        for index, node in self.out_nodes.items():
            node.set_value(self.a[index].compute(self.o[self.a[index].head_node].value))

    def update_gradient(self, y=None):
        """更新层内节点的梯度"""
        gradient = defaultdict(np.float64)
        gradient_weight = defaultdict(np.float64)
        for index, weight_edge in self.w.items():
            gradient[index] = weight_edge.diff_local() * self.a[index[1]].diff_local(self.o[index[1]].value)
            gradient_weight[index] = self.x[index[0]].value * self.a[index[1]].diff_local(self.o[index[1]].value)
        for index, g in gradient.items():
            block_index = index[-1]
            self.gradient[block_index][index] = g
        self.gradient_weight = gradient_weight


class SoftmaxLossLayer:
    """loss和softmax的复合"""

    def __init__(self, no, last):
        self.no = no
        self.x = last
        n = len(self.x)
        self.out_nodes = {0: Node(self.no, 0, "value_softmax")}

        # softmax-loss边的字典
        self.s = {}
        for i in range(n):
            edge = SoftmaxLossEdge(-1, i,
                                   head_node=i,
                                   tail_node=-1)
            self.s[i] = edge

        self.gradient = defaultdict(dict)

    def forward(self, y):
        """前向传播"""
        self.all_x = [node.value for node in self.x.values()]
        loss_value = self.s[0].compute(self.all_x, y)
        self.out_nodes = {0: Node(-1, 0, "loss", loss_value)}

    def get_out(self):
        """获取输出，得到softmax后的概率分布"""
        self.all_x = [node.value for node in self.x.values()]
        return self.s[0].compute2(self.all_x)

    def update_gradient(self, y):
        """更新梯度"""
        gradient = defaultdict(np.float64)
        for index, weight_edge in self.s.items():
            gradient[index] = weight_edge.diff_local(self.all_x, y, index)
        for index, g in gradient.items():
            self.gradient[0][(index, 0)] = g


class SoftmaxLayer:
    """单独的softmax层（暂时没有用到）"""

    def __init__(self, last):
        self.x = last
        n = len(self.x)
        self.out_nodes = {0: Node(-1, 0, "value_softmax")}
        self.s = {}
        for i in range(n):
            edge = LossEdge(-1, 1,
                            head_node=i,
                            tail_node=0)
            self.s[(i, 0)] = edge


class Model:
    def __init__(self, n_input, n_nodes: list):
        assert n_input > 0
        assert len(n_nodes) > 0
        for _ in n_nodes:
            assert _ > 0

        self.n_input = n_input
        self.n_nodes = n_nodes

        self.n_layers = len(n_nodes)
        self.layers = []

    def build(self):
        """建立模型"""
        n_in = self.n_input
        last = {i: Node(1, i, "value_activated") for i in range(n_in)}
        for i, n in enumerate(self.n_nodes):
            n_out = n
            layer = FullyConnectedLayer(i, n_in, n_out, last)
            last = layer.out_nodes
            self.layers.append(layer)
            n_in = n_out
        # 最后一层为softmax-loss层
        layer = SoftmaxLossLayer(len(self.n_nodes), last)
        self.layers.append(layer)

    def fp(self, x, y):
        """模型的前向传播"""
        assert len(x) == len(self.layers[0].x)
        n = len(x)
        layer_out = {i: Node(1, i, "value_activated", x[i]) for i in range(n)}
        self.layers[0].x = layer_out
        for i, layer in enumerate(self.layers[:-1]):
            # 每一层前向传播，并把上一层的输出作为本层输入
            layer.x = layer_out
            layer.forward()
            layer_out = layer.out_nodes

        softmax_loss_layer = self.layers[-1]
        softmax_loss_layer.x = layer_out
        softmax_loss_layer.forward(y)

    def get_loss(self):
        """获取loss的值"""
        return self.layers[-1].out_nodes[0].value

    def predict(self):
        """预测，返回概率分布"""
        return self.layers[-1].get_out()

    def get_weight(self):
        """获取模型中的权重"""
        weight = {}
        for layer in self.layers[:-1]:
            for index, w in layer.w.items():
                weight[(layer.no, *index)] = w.weight
        return weight

    def bp(self, x, y):
        """反向穿壁"""
        self.fp(x, y)
        layers_gradient = {}
        gradient_paths = []
        gradient_weight = {}
        for layer in self.layers[::-1]:
            # 每一层更新梯度
            index_layer = layer.no
            layer.update_gradient(y)

            layers_gradient[index_layer] = layer.gradient
            if index_layer != len(self.layers) - 1:
                gradient_weight[index_layer] = layer.gradient_weight

            for tail, index_g in layer.gradient.items():
                for index, g in index_g.items():
                    if index_layer == len(self.layers) - 1:
                        gradient_paths.append(Path(index_layer, [index], g))
                    else:
                        tmp = []

                        for gradient_path in gradient_paths:
                            if gradient_path.head == index[1] and gradient_path.layer == index_layer + 1:
                                tmp.append(Path(index_layer, [index] + gradient_path.path, gradient_path.g * g))
                        gradient_paths.extend(tmp)
        # 获取每个对应到每个权重更新是需要用到的梯度
        gradients = {}
        for layer in self.layers[:-1]:
            index_layer = layer.no
            for v in layer.gradient.values():
                for index in v.keys():
                    delta = gradient_weight[index_layer][index] * np.sum(
                        [path.g for path in gradient_paths if path.layer == index_layer + 1 and path.head == index[1]])
                    gradients[(index_layer, *index)] = delta
        return gradients

    def update(self, gradients, lr, gamma=0):
        """更新模型权重"""
        for layer in self.layers[:-1]:
            index_layer = layer.no
            for v in layer.gradient.values():
                for index in v.keys():
                    delta = gradients[(index_layer, *index)]
                    layer.w[index].update_weight(delta, alpha=lr, gamma=gamma)


def find_all_paths(layer, head, tail, gradient_paths):
    """在计算图中找到以head为首，以tail为终点的所有路径"""
    return [path for path in gradient_paths if path.layer == layer and path.head == head and path.tail == tail]


def collect_grad(list_grad):
    """将所有单个样本计算得到的梯度取平均，用于batch训练"""
    result = {}
    for grad in list_grad:
        for index, g in grad.items():
            if index not in result:
                result[index] = g / len(list_grad)
            else:
                result[index] += g / len(list_grad)
    return result


class Path:
    """计算图中的一条路径"""

    def __init__(self, layer, path, g):
        self.layer = layer
        self.path = path
        # 路径上偏导数的乘积（链式法则）
        self.g = g
        self.head = self.path[0][0]
        self.tail = self.path[-1][-1]

    def append(self, p, g):
        """延长路径，更新g"""
        self.path.append(p)
        self.g *= g
        self.tail = self.path[-1][-1]


def get_model(len_input, len_output, shape):
    """获取模型"""
    model = Model(len_input, shape)
    model.build()
    return model


def train_model(model, data, split_rate=0.8, iterations=1000, lr=0.01, gamma=0, batch_size=16):
    """训练模型"""
    data_train, data_test = data[: int(split_rate * len(data))], data[int(split_rate * len(data)):]
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    train_result = []
    test_result = []
    train_tmp_loss = []
    test_tmp_loss = []
    for i in tqdm(range(iterations)):
        grads = []
        for x, y in random.sample(data_train, k=batch_size):
            model.fp(x, y)
            train_tmp_loss.append(model.get_loss())
            grads.append(model.bp(x, y))
            train_result.append(np.array(model.predict()).argmax() == np.array(y).argmax())
        for x, y in data_test:
            model.fp(x, y)
            test_tmp_loss.append(model.get_loss())
            test_result.append(np.array(model.predict()).argmax() == np.array(y).argmax())
        if i:
            train_loss.append(np.mean(train_tmp_loss))
            train_acc.append(len([r for r in train_result if r]) / len(train_result))
            test_loss.append(np.mean(test_tmp_loss))
            test_acc.append(len([r for r in test_result if r]) / len(test_result))
            print(f"train loss:\t{train_loss[-1]}", end='\t')
            print(f"train acc:\t{train_acc[-1]}", end='\t')
            print(f"test loss:\t{test_loss[-1]}", end='\t')
            print(f"test acc:\t{test_acc[-1]}")

        model.update(collect_grad(grads), lr=lr, gamma=gamma)
    return train_loss, train_acc, test_loss, test_acc


def get_data():
    """获取鸢尾花数据"""
    data = load_iris()
    enc = OneHotEncoder()
    data_x = scale([d for d in data.data], axis=0, with_mean=True, with_std=True, copy=True)
    data_x = [np.array(d.tolist() + [1]) for d in data_x]
    data_y = enc.fit_transform(data.target.reshape(-1, 1)).toarray()
    data = list(zip(data_x, data_y))
    random.shuffle(data)
    return data


def run(conf):
    """pipeline"""
    lr = conf['lr']
    gamma = conf['gamma']
    batch_size = conf['batch_size']
    hidden_nodes = conf['hidden_nodes']

    len_input = 4 + 1  # 增广1
    len_output = 3  # onehot
    n_nodes = [hidden_nodes, len_output]
    model = get_model(len_input, len_output, n_nodes)
    data = get_data()
    history = train_model(model, data, lr=lr, gamma=gamma, batch_size=batch_size)
    return history


if __name__ == '__main__':
    # 比较学习率（步长）
    configures_1 = [
        {
            "lr": lr,
            "gamma": 0,
            "batch_size": 16,
            "hidden_nodes": 8
        }
        for lr in [0.01, 0.05, 0.1, 0.5, 1, 5]
    ]

    # 比较隐藏层节点
    configures_2 = [
        {
            "lr": 0.1,
            "gamma": 0,
            "batch_size": 16,
            "hidden_nodes": n
        }
        for n in [1, 4, 16, 64]
    ]

    # 比较gamma(动量参数)
    configures_3 = [
        {
            "lr": 0.1,
            "gamma": g,
            "batch_size": 16,
            "hidden_nodes": 8
        }
        for g in [0, 0.9, 0.7, 0.5, 0.3, 0.1]
    ]
    # 比较batch size
    configures_4 = [
        {
            "lr": 0.1,
            "gamma": 0,
            "batch_size": bs,
            "hidden_nodes": 8
        }
        for bs in [1, 2, 4, 8, 16, 32, 64]
    ]

    for i, configures in enumerate([configures_1, configures_2, configures_3, configures_4]):
        for conf in configures:
            history = run(conf)
            with open(f"pkl_files/{i}-{str(conf)}.pkl", 'wb') as f:
                pickle.dump(history, f)
