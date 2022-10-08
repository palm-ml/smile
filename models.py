from functools import total_ordering
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torchvision.models import resnet50
from collections import OrderedDict
import copy

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
    
    def __init__(self, P, observed_label_matrix, estimated_labels):
        
        super(LabelEstimator, self).__init__()
        print('initializing label estimator')
        
        # Note: observed_label_matrix is assumed to have values in {-1, 0, 1} indicating 
        # observed negative, unknown, and observed positive labels, resp.
        
        num_examples = int(np.shape(observed_label_matrix)[0])
        observed_label_matrix = np.array(observed_label_matrix).astype(np.int8)
        total_pos = np.sum(observed_label_matrix == 1)
        total_neg = np.sum(observed_label_matrix == -1)
        print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
        print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))
        
        if estimated_labels is None:
            # initialize unobserved labels:
            w = 0.1
            q = inverse_sigmoid(0.5 + w)
            param_mtx = q * (2 * torch.rand(num_examples, P['num_classes']) - 1)
            
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
    def __init__(self, P):
        super(Decoder, self).__init__()
        self.arch = P['arch']
        if self.arch == 'linear':
            self.linear_classifier = FCNet(P['num_classes'], P['feat_dim'])
        elif self.arch == 'mlp':
            self.linear_classifier = MLPNet(P['num_classes'], P['feat_dim'])
    
    def forward(self, x):
        m = torch.distributions.Beta(x + 1e-5, 1 - x.clone() + 1e-5)
        rec_y = m.rsample()
        rec_x = self.linear_classifier(rec_y)
        rec_A = torch.mm(rec_y, rec_y.T)
        return rec_x, rec_y, rec_A

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
