import math
from typing_extensions import Self
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torchvision.models import resnet50
import copy
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from dgl.nn.pytorch import GraphConv

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size, num_classes, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.fc = torch.nn.Linear(num_classes, num_classes)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = self.fc(h)
        return h


'''
utility functions
'''

def inverse_sigmoid(p):
    epsilon = 1e-5
    p = np.minimum(p, 1 - epsilon)
    p = np.maximum(p, epsilon)
    return np.log(p / (1-p))

'''
model definitions
'''

class MLPNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(MLPNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_feats, num_classes)
        self.fc2 = torch.nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(FCNet, self).__init__()
        self.fc = torch.nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# # 定义图卷积层
# class GraphConvolution(Module):

#     # 图卷积层的作用是接收旧特征并产生新特征
#     # 因此初始化的时候需要确定两个参数：输入特征的维度与输出特征的维度
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         #parameter 作用是将tensor设置为梯度求解，并将其绑定到模型的参数中。
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     # 参数的初始化
#     def reset_parameters(self):

#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
    
#     # 定义前向计算（邻居聚合与特征变换）
#     # 输入是旧特征+邻接矩阵
#     # 输出是新特征
#     def forward(self, input, adj):

#         # 特征变换
#         support = torch.mm(input, self.weight)
#         # 邻居聚合
#         output = torch.spmm(adj, support)

#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#     # 方法是类的实例化对象用来做“自我介绍”的方法，
#     # 默认情况下，它会返回当前对象的“类名+object at+内存地址”， 
#     # 而如果对该方法进行重写，可以为其制作自定义的自我描述信息。
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

# class GCN(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout=0.1):
#         super(GCN, self).__init__()

#         #这里其实可以使用列表将其收集，即可定义包含任意图卷积层的GCN
#         #第一层GCN
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         #第二层GCN
#         self.gc2 = GraphConvolution(nhid, nclass)

#         self.dropout = dropout
    
#     #定义前向计算，即各个图卷积层之间的计算逻辑
#     def forward(self, x, adj):
#         #第一层的输出
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         #第二层的输出
#         x = self.gc2(x, adj)
#         return x


class ImageClassifier(torch.nn.Module):
    
    def __init__(self, P, model_feature_extractor=None, model_linear_classifier=None):
        
        super(ImageClassifier, self).__init__()
        print('initializing image classifier')
        
        model_feature_extractor_in = copy.deepcopy(model_feature_extractor)
        model_linear_classifier_in = copy.deepcopy(model_linear_classifier)
        
        self.arch = P['arch']
        
        if self.arch == 'resnet50':
            # configure feature extractor:
            if model_feature_extractor_in is not None:
                print('feature extractor: specified by user')
                feature_extractor = model_feature_extractor_in
            else:
                if P['use_pretrained']:
                    print('feature extractor: imagenet pretrained')
                    feature_extractor = resnet50(pretrained=True)
                else:
                    print('feature extractor: randomly initialized')
                    feature_extractor = resnet50(pretrained=False)
                feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            if P['freeze_feature_extractor']:
                print('feature extractor frozen')
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                print('feature extractor trainable')
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1) 
            self.feature_extractor = feature_extractor
            
            # configure final fully connected layer:
            if model_linear_classifier_in is not None:
                print('linear classifier layer: specified by user')
                linear_classifier = model_linear_classifier_in
            else:
                print('linear classifier layer: randomly initialized')
                linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
            self.linear_classifier = linear_classifier
            
        elif self.arch == 'linear':
            print('training a linear classifier only')
            self.feature_extractor = None
            self.linear_classifier = FCNet(P['feat_dim'], P['num_classes'])
        elif self.arch == 'mlp':
            print('training a mlp classifier only')
            self.feature_extractor = None
            self.linear_classifier = MLPNet(P['feat_dim'], P['num_classes'])
        else:
            raise ValueError('Architecture not implemented.')
    
    def forward(self, x):
        if self.arch == 'linear' or self.arch == 'mlp':
            # x is a batch of feature vectors
            logits = self.linear_classifier(x)
        else:
            # x is a batch of images
            feats = self.feature_extractor(x)
            logits = self.linear_classifier(torch.squeeze(feats))
        return logits

class LabelEstimator(torch.nn.Module):
    
    def __init__(self, observed_label_matrix, estimated_labels):
        
        super(LabelEstimator, self).__init__()
        print('initializing label estimator')
        
        # Note: observed_label_matrix is assumed to have values in {-1, 0, 1} indicating 
        # observed negative, unknown, and observed positive labels, resp.
        
        num_examples = int(np.shape(observed_label_matrix)[0])
        num_classes  = int(np.shape(observed_label_matrix)[1])
        observed_label_matrix = np.array(observed_label_matrix).astype(np.int8)
        total_pos = np.sum(observed_label_matrix == 1)
        total_neg = np.sum(observed_label_matrix == -1)
        print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
        print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))
        
        if estimated_labels is None:
            # initialize unobserved labels:
            w = 0.1
            q = inverse_sigmoid(0.5 + w)
            param_mtx = q * (2 * torch.rand(num_examples, num_classes) - 1)
            
            # initialize observed positive labels:
            init_logit_pos = inverse_sigmoid(0.995)
            idx_pos = torch.from_numpy((observed_label_matrix == 1).astype(np.bool))
            param_mtx[idx_pos] = init_logit_pos
            # initialize observed negative labels:
            init_logit_neg = inverse_sigmoid(0.005)
            idx_neg = torch.from_numpy((observed_label_matrix == -1).astype(np.bool))
            param_mtx[idx_neg] = init_logit_neg
        else:
            param_mtx = inverse_sigmoid(torch.FloatTensor(estimated_labels))
        
        self.logits = torch.nn.Parameter(param_mtx)
        
    def get_estimated_labels(self):
        with torch.set_grad_enabled(False):
            estimated_labels = torch.sigmoid(self.logits)
        estimated_labels = estimated_labels.clone().detach().cpu().numpy()
        return estimated_labels
    
    def forward(self, indices):
        x = self.logits[indices, :]
        x = torch.sigmoid(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_classes, hid_dim, feat_dim):
        super(Decoder, self).__init__()
        self.linear_classifier = MLPNet(num_classes + hid_dim, feat_dim)
    
    def forward(self, distribution, hid_feat):
        m = torch.distributions.Beta(distribution + 1e-5, 1 - distribution.clone() + 1e-5)
        rec_y = m.rsample()
        rec_x = self.linear_classifier(torch.cat((distribution, hid_feat), dim=1))
        rec_A = torch.mm(rec_y, rec_y.T)
        return rec_x, rec_y, rec_A


class Encoder_SMILE(torch.nn.Module):
    def __init__(self, feat_dim, num_classes, z_dim):
        super(Encoder_SMILE, self).__init__()
        hidden_dim = int(feat_dim / 2)
        self.fc1 = torch.nn.Linear(feat_dim, hidden_dim)
        self.log_alpha = torch.nn.Linear(hidden_dim, num_classes)
        self.log_beta = torch.nn.Linear(hidden_dim, num_classes)
        self.fc2 = torch.nn.Linear(feat_dim + num_classes, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = torch.nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # encoder d
        feat = self.fc1(x)
        feat = F.relu(feat)
        log_alpha = self.log_alpha(feat)
        log_beta = self.log_beta(feat)
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        d_sampler = torch.distributions.Beta(alpha, beta)
        d = d_sampler.rsample()
        # encoder z
        feat = self.fc2(x)
        feat = F.relu(feat)
        mu = self.fc_mu(feat)
        logstd = self.fc_logstd(feat)
        std = torch.exp(logstd)
        z_sampler = torch.distributions.Normal(mu, std)
        z = z_sampler.rsample()

        return d, z, alpha, beta, mu, std


class Decoder_SMILE(torch.nn.Module):
    def __init__(self, feat_dim, num_classes, z_dim):
        super(Decoder_SMILE, self).__init__()
        self.decoder_x = MLPNet(num_classes + z_dim, feat_dim)
        self.decoder_y = MLPNet(num_classes, num_classes)

    def forward(self, logits_label, z):
        rec_x = self.decoder_x(torch.cat((logits_label, z), dim=1))
        rec_y = self.decoder_y(logits_label)
        rec_y = F.sigmoid(rec_y)
        return rec_x, rec_y

class MultilabelModel(torch.nn.Module):
    def __init__(self, P, feature_extractor, linear_classifier, observed_label_matrix, estimated_labels=None):
        super(MultilabelModel, self).__init__()
        
        self.f = ImageClassifier(P, feature_extractor, linear_classifier)
        
        self.g = LabelEstimator(P, observed_label_matrix, estimated_labels)

        self.d = Decoder(P)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        g_preds = self.g(batch['idx']) # oops, we had a sigmoid here in addition to 
        return (f_logits, g_preds)


class MultilabelContrastModel(torch.nn.Module):
    def __init__(self, P, feature_extractor, linear_classifier):
        super(MultilabelContrastModel, self).__init__()
        
        self.f = ImageClassifier(P, feature_extractor, linear_classifier)
        
        self.g = ImageClassifier(P, feature_extractor, linear_classifier)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        g_logits = self.g(batch['image']) # oops, we had a sigmoid here in addition to 
        return (f_logits, g_logits)
