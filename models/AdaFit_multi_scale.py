import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import normal_estimation_utils
import ThreeDmFVNet
from pytorch3d.ops import knn_points, knn_gather

from netBase import BaseBC,BaseBF
import math


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx



def atten_get_graph_feature(x, k=20, idx=None, firstlayer=False, p=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x_ver = x.unsqueeze(-1)
    x_glo1 = torch.sum(x, dim=1)

    x_glo2 = torch.mean(x, dim=1)
    x_glo = torch.cat((x_glo1, x_glo2), dim=-1)

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    DEd1 = feature - x
    DEd2 = DEd1.mul(DEd1)
    DEd3 = torch.sum(DEd2, dim=-1, keepdim=True)

    if firstlayer == True:  # HPE strategy
        feature = torch.cat((DEd3, feature - x, x, feature), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    if p != None:
        p = p[:, :, :num_points]

        mlp_pos = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        ).to(device)

        _, p_dims, _ = p.size()
        p = p.view(batch_size, -1, num_points)
        p = p.transpose(2, 1).contiguous()
        points = p.view(batch_size * num_points, -1)[idx, :]
        points = points.view(batch_size, num_points, k, p_dims) # (B,N,k,3)
        p = p.view(batch_size, num_points, 1, p_dims).repeat(1, 1, k, 1) # (B,N,k,3)

        DEd1 = points - p

        encoding_feat = mlp_pos(DEd1)  # (B,n,K,64)
        encoding_feat = torch.cat([DEd1, encoding_feat], dim=-1)  # (B,n,K,3+64)
        
        p = encoding_feat.permute(0, 3, 1, 2).contiguous()

        '''
        DEd2 = DEd1.mul(DEd1)
        DEd3 = torch.sum(DEd2, dim=-1, keepdim=True)
        p = torch.cat((points - p, p, points, DEd3), dim=3).permute(0, 3, 1, 2).contiguous()
        '''
    return x_ver, feature, x_glo, p


class PointNetFeatures(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', k=20):
        super(PointNetFeatures, self).__init__()
        self.k = k
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales
        self.conv1 = torch.nn.Conv1d(3, 64, 1)

        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)


        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=500*self.point_tuple, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=500, dim=64, sym_op=self.sym_op)

        self.graph_attention1 = LAB(3, 10, 64)

    def forward(self, x):
        n_pts = x.size()[2]
        points = x
        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            #x = x.view(x.size(0), 3, -1)
           
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3 * self.point_tuple, -1)
            points = x
        else:
            trans = None

        x_ver1, x_fea1, x_glo, _ = atten_get_graph_feature(points, k=self.k, firstlayer=True)  # x_glo:(256,512)
        x1 = self.graph_attention1(x_ver1, x_fea1)

        #x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x1)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        return x, x1, x_glo, trans, trans2, points


class VariableScaleLayer(nn.Module):
    def __init__(self, input_scale,output_scale,input_dim,output_dim,need_pre_global=False):
        #num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max'):
        super(VariableScaleLayer, self).__init__()
        #self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         # use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op)
        self.input_scale = input_scale
        self.input_dim=input_dim
        self.output_scale = output_scale
        self.output_dim = output_dim
        self.pre_bc = BaseBC(input_dim,input_dim*2)
        self.fc = BaseBF(input_dim*2,input_dim//2)
        self.need_pre_global = need_pre_global
        if need_pre_global:
            self.bc = BaseBC(input_dim+input_dim//2 + input_dim//4,output_dim)
        else:
            self.bc = BaseBC(input_dim+input_dim//2,output_dim)

    def forward(self, x):

        x,pre_global_feature = x
        batch_size =  x.size()[0]
        
        global_feature = torch.max(self.pre_bc(x), 2, keepdim=False)[0]
        global_feature = self.fc(global_feature)
        if self.need_pre_global:
            x = torch.cat([x[:,:,:self.output_scale], global_feature.view(batch_size, -1, 1).repeat(1, 1, self.output_scale),pre_global_feature.view(batch_size, -1, 1).repeat(1, 1, self.output_scale)], 1)
        else:    
            x = torch.cat([x[:,:,:self.output_scale], global_feature.view(batch_size, -1, 1).repeat(1, 1, self.output_scale)], 1)
        x = self.bc(x)
        return x,global_feature


class LAB(nn.Module):  # Local Attention Block
    def __init__(self, size1, size2, size3):
        super(LAB, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3

        self.bn1 = nn.BatchNorm2d(self.size3)
        self.bn2 = nn.BatchNorm2d(self.size3)
        self.bn3 = nn.BatchNorm2d(1)

        self.keyconv = nn.Sequential(nn.Conv2d(self.size1, self.size3, kernel_size=1, bias=False),
                                     # The query vector of attetion
                                     self.bn1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.valconv = nn.Sequential(nn.Conv2d(self.size2, self.size3, kernel_size=1, bias=False),
                                     # The value vector of attention
                                     self.bn2,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.scoconv = nn.Sequential(nn.Conv2d(self.size3, 1, kernel_size=1, bias=False),  # Attention scores
                                     self.bn3,
                                     nn.LeakyReLU(negative_slope=0.2))



    def forward(self, x_query, x_key):
        querys, values = self.keyconv(x_query), self.valconv(x_key)
        features = querys + values
        scores = self.scoconv(features).squeeze(1)
        scores = F.softmax(scores, dim=2) # vis scores.sum(2).unsqueeze(1)
        scores = scores.unsqueeze(1).repeat(1, self.size3, 1, 1) # (B,C,N,k)
        feature = values.mul(scores)
        feature = torch.sum(feature, dim=-1)

        return feature


class GAB(nn.Module):  # Global Attention Block
    def __init__(self):
        super(GAB, self).__init__()
        self.bn1 = nn.BatchNorm1d(250)
        #self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(1)

        self.Linear1 = nn.Linear(256, 512, bias=False)
        self.Linear2 = nn.Linear(512, 256, bias=False)
        self.Linear3 = nn.Linear(256, 64, bias=False)
        self.Linear4 = nn.Linear(64, 1, bias=False)
        self.conv1 = nn.Conv1d(125, 250, 1, bias=False)
        #self.conv1 = nn.Conv1d(175, 1400, 1, bias=False)
        #self.conv2 = nn.Conv1d(1000, 256, 1, bias=False)
        #self.conv2 = nn.Conv1d(1400, 175, 1, bias=False)
        self.conv3 = nn.Conv1d(250, 64, 1, bias=False)
        #self.conv3 = nn.Conv1d(175, 64, 1, bias=False)
        self.conv4 = nn.Conv1d(64, 1, 1, bias=False)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.dp3 = nn.Dropout(p=0.5)
        self.dp4 = nn.Dropout(p=0.5)


    def forward(self, x_query, x_key): # (B,512,256) (B,256)
        n_pts = x_key.size()[1]
        #values = F.leaky_relu(self.bn1(self.Linear1(x_key).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        values = F.leaky_relu(self.bn1(self.conv1(x_key)), negative_slope=0.2)
        values = self.dp1(values)  # The value vector (B,256,2048) (B,1000,512)
        #querys = x_query.unsqueeze(1)  # The query vector (B,1,512) (B,1,256)
        querys = x_query.unsqueeze(-1)  # The query vector (B,1,512) (B,1000,1)
        #querys = F.leaky_relu(self.bn(self.Linear(querys).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        #querys = self.dp(querys)
        features = values + querys # (B,256,256) (B,1000,512)
        #scores = F.leaky_relu(self.bn2(self.Linear2(features).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        #scores = F.leaky_relu(self.bn2(self.conv2(features)), negative_slope=0.2) # (B,256,512)
        #scores = self.dp2(scores)
        #scores = F.leaky_relu(self.bn3(self.Linear3(scores).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        scores = F.leaky_relu(self.bn3(self.conv3(features)), negative_slope=0.2) # (B,64,512)
        scores = self.dp3(scores)
        #scores = F.leaky_relu(self.bn4(self.Linear4(scores).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        scores = F.leaky_relu(self.bn4(self.conv4(scores)), negative_slope=0.2) # (B,1,512)
        scores = self.dp4(scores)  # Attention score
        #scores = scores.squeeze(-1)
        scores = scores.squeeze(1) # (B,512)
        scores = F.softmax(scores, dim=1) # (B,1)
        scores = scores.unsqueeze(-1).repeat(1, 1, 2*n_pts) # (B,1,1000)
        #scores = scores.unsqueeze(1).repeat(1, 512, 1) # (B,512,1)
        #scores = scores.transpose(2, 1).repeat(1, 1, 512)  # (B,1,512)
        feature = values.mul(scores.transpose(2, 1)) # (B,1000,512)
        feature = torch.sum(feature, dim=1, keepdim=True)  # (B,1,512)
        feature = feature.transpose(2, 1).contiguous().repeat(1, 1, n_pts)  # (B,768,N/4)
        return feature

class PointNetEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', k=20):
        super(PointNetEncoder, self).__init__()
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op)
        self.k = k
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales

        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.vsl_1 = VariableScaleLayer(num_points,num_points//2,128,256)
        self.vsl_2 = VariableScaleLayer(num_points//2,num_points//4,256,256,need_pre_global=True)

        self.conv3 = torch.nn.Conv1d(256, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


        self.graph_attention2 = LAB(64, 64*2+64+3 , 64)
        self.graph_attention3 = LAB(128, 128*2+64+3 , 128)
        self.graph_attention4 = LAB(256, 256*2+64+3 , 256)
        self.graph_attention5 = LAB(256, 256*2+64+3 , 256)
        self.GlobalAtten = GAB()

        self.mlp_pos = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            )

    def forward(self, points):
        n_pts = points.size()[2]
        #pointfeat, trans, trans2, points = self.pointfeat(points)
        pointfeat, x1, x_glo, trans, trans2, points = self.pointfeat(points)

        x_ver2, x_fea2, _, pos_enc2 = atten_get_graph_feature(pointfeat, k=self.k, p=points)
        x_fea2 = torch.cat([x_fea2, pos_enc2], 1)
        x2 = self.graph_attention2(x_ver2, x_fea2)

        x = F.relu(self.bn2(self.conv2(x2)))

        x_ver3, x_fea3, _, pos_enc3 = atten_get_graph_feature(x, k=self.k, p=points)
        x_fea3 = torch.cat([x_fea3, pos_enc3], 1)
        x3 = self.graph_attention3(x_ver3, x_fea3)

        x,global_500 = self.vsl_1([x3,None]) # 500->250

        x_ver4, x_fea4, _, pos_enc4 = atten_get_graph_feature(x, k=self.k, p=points) # 250
        x_fea4 = torch.cat([x_fea4, pos_enc4], 1)
        x4 = self.graph_attention4(x_ver4, x_fea4)

        x,global_256 = self.vsl_2([x4,global_500]) #  250->125

        x_ver5, x_fea5, idx, pos_enc5 = atten_get_graph_feature(x, k=self.k, p=points)
        x_fea5 = torch.cat([x_fea5, pos_enc5], 1)
        x5 = self.graph_attention5(x_ver5, x_fea5)

        x_LAB = torch.cat((x1[:,:,:n_pts//4], x2[:,:,:n_pts//4], x3[:,:,:n_pts//4], x4[:,:,:n_pts//4], x5), dim=1)  # (B,768,N/4)
        global_LAB = torch.max(x_LAB, 2, keepdim=True)[0]
        global_LAB = global_LAB.repeat(1, 1, n_pts//4)  # (B,768,N/4)
        x_LAB = x_LAB.transpose(2, 1).contiguous()  # (B,N/4,768)

        #######positional_encoding######
        pos = points.transpose(1, 2).contiguous()
        _, knn_idx, _ = knn_points(pos, pos, K=17, return_nn=False)  # (B, N, K+1)

        pos_sub = pos[:, :n_pts//4, :]  # (B, n, 3)
        knn_idx = knn_idx[:, :n_pts//4, :16]  # (B, n, K)

        nn_pc = knn_gather(pos, knn_idx)  # (B, n, K, 3)
        #nn_pc = nn_pc - pos_sub.unsqueeze(2)  # (B, n, K, 3)

        pos_enc = self.mlp_pos(nn_pc)  # (B, n, K, C)
        pos_enc = torch.max(pos_enc, dim=2, keepdim=False)[0]
        pos_enc = pos_enc.transpose(1, 2).contiguous() # (B, C, n)
        ################################
        shape_sum = torch.sum(pos_enc, dim=1)
        shape_mean = torch.mean(pos_enc, dim=1)
        shape_descriptor = torch.cat((shape_sum, shape_mean), dim=-1) # (B, 2*n)
        x_GAB = self.GlobalAtten(shape_descriptor, x_LAB)

        pointfeat = x5
        x = self.bn3(self.conv3(pointfeat))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, n_pts//4)
        points = points[:, :, :n_pts // 4]
        #return torch.cat([x, pointfeat], 1), global_feature.squeeze(), trans, trans2, points
        return torch.cat([global_feature, x_GAB, global_LAB, pointfeat], 1), global_feature.squeeze(), trans, trans2, points


class PointNet3DmFVEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', n_gaussians=5):
        super(PointNet3DmFVEncoder, self).__init__()
        self.num_points = num_points
        self.point_tuple = point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales = num_scales
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op)

        self.n_gaussians = n_gaussians

        self.gmm = ThreeDmFVNet.get_3d_grid_gmm(subdivisions=[self.n_gaussians, self.n_gaussians, self.n_gaussians],
                              variance=np.sqrt(1.0 / self.n_gaussians))


    def forward(self, x):
        points = x
        n_pts = x.size()[2]

        pointfeat, trans, trans2, points = self.pointfeat(points)
        global_feature = ThreeDmFVNet.get_3DmFV_pytorch(points.permute([0, 2, 1]), self.gmm.weights_, self.gmm.means_,
                                              np.sqrt(self.gmm.covariances_), normalize=True)
        global_feature = torch.flatten(global_feature, start_dim=1)
        x = global_feature.unsqueeze(-1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), global_feature.squeeze(), trans, trans2, points


class DeepFit(nn.Module):
    def __init__(self, k=1, num_points=500, use_point_stn=False,  use_feat_stn=False, point_tuple=1,
                 sym_op='max', arch=None, n_gaussians=5, jet_order=2, weight_mode="tanh",
                 use_consistency=False):
        super(DeepFit, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.num_points=num_points
        self.point_tuple = point_tuple
        if arch == '3dmfv':
            self.n_gaussians = n_gaussians  # change later to get this as input
            self.feat = PointNet3DmFVEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                        point_tuple=point_tuple, sym_op=sym_op, n_gaussians= self.n_gaussians )
            feature_dim = self.n_gaussians * self.n_gaussians * self.n_gaussians * 20 + 64
        else:
            self.feat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                            point_tuple=point_tuple, sym_op=sym_op)

            feature_dim = 1024 + 768 + 768 + 256
        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.conv_bias = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.jet_order = jet_order
        self.weight_mode = weight_mode
        self.compute_neighbor_normals = use_consistency
        self.do = torch.nn.Dropout(0.25)


    def forward(self, points):

        x, _, trans, trans2, points = self.feat(points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        bias = self.conv_bias(x)
        bias[:,:,0] = 0
        points = points + bias
        # point weight estimation.
        if self.weight_mode == "softmax":
            x = F.softmax(self.conv4(x))
            weights = 0.01 + x  # add epsilon for numerical robustness
        elif self.weight_mode =="tanh":
            x = torch.tanh(self.conv4(x))
            weights = (0.01 + torch.ones_like(x) + x) / 2.0  # learn the residual->weights start at 1
        elif self.weight_mode =="sigmoid":
            weights = 0.01 + torch.sigmoid(self.conv4(x))

        beta, normal, neighbor_normals = fit_Wjet(points, weights.squeeze(), order=self.jet_order,
                                                              compute_neighbor_normals=self.compute_neighbor_normals)

        return normal, beta.squeeze(), weights.squeeze(), trans, trans2, neighbor_normals, bias, points


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.vsl_1 = VariableScaleLayer(700,350,128,256)
        self.vsl_2 = VariableScaleLayer(350,175,256,128)
        #self.bc = BaseBC(256,128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(175)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x,global_500 = self.vsl_1([x,None])
        x,global_256 = self.vsl_2([x,global_500])
        #x = self.bc(x)
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.vsl_1 = VariableScaleLayer(700,350,128,256)
        self.vsl_2 = VariableScaleLayer(350,175,256,128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(175)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x,global_500 = self.vsl_1([x,None])
        x,global_256 = self.vsl_2([x,global_500])
        #x = self.bc(x)
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = normal_estimation_utils.batch_quat_to_rotmat(x)

        return x


def fit_Wjet(points, weights, order=2, compute_neighbor_normals=False):
    """
    Fit a "n-jet" (n-order truncated Taylor expansion) to a point clouds with weighted points.
    We assume that PCA was performed on the points beforehand.
    To do a classic jet fit input weights as a one vector.
    :param points: xyz points coordinates
    :param weights: weight vector (weight per point)
    :param order: n-order of the jet
    :param compute_neighbor_normals: bool flag to compute neighboring point normal vector

    :return: beta: polynomial coefficients
    :return: n_est: normal estimation
    :return: neighbor_normals: analytically computed normals of neighboring points
    """

    neighbor_normals = None
    batch_size, D, n_points = points.shape

    # compute the vandermonde matrix
    x = points[:, 0, :].unsqueeze(-1)
    y = points[:, 1, :].unsqueeze(-1)
    z = points[:, 2, :].unsqueeze(-1)
    weights = weights.unsqueeze(-1)

    # handle zero weights - if all weights are zero set them to 1

    valid_count = torch.sum(weights > 1e-3, dim=1)
    w_vector = torch.where(valid_count > 18, weights.view(batch_size, -1),
                            torch.ones_like(weights, requires_grad=True).view(batch_size, -1)).unsqueeze(-1)

    if order > 1:
        #pre conditioning
        h = (torch.mean(torch.abs(x), 1) + torch.mean(torch.abs(y), 1)) / 2 # absolute value added from https://github.com/CGAL/cgal/blob/b9e320659e41c255d82642d03739150779f19575/Jet_fitting_3/include/CGAL/Monge_via_jet_fitting.h
        # h = torch.mean(torch.sqrt(x*x + y*y), dim=2)
        idx = torch.abs(h) < 0.0001
        h[idx] = 0.1
        # h = 0.1 * torch.ones(batch_size, 1, device=points.device)
        x = x / h.unsqueeze(-1).repeat(1, n_points, 1)
        y = y / h.unsqueeze(-1).repeat(1, n_points, 1)

    if order == 1:
        A = torch.cat([x, y, torch.ones_like(x)], dim=2)
    elif order == 2:
        A = torch.cat([x, y, x * x, y * y, x * y, torch.ones_like(x)], dim=2)
        h_2 = h * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, torch.ones_like(h)], dim=1))
    elif order == 3:
        y_2 = y * y
        x_2 = x * x
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_2 * x, y_2 * y, x_2 * y, y_2 * x,  torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, torch.ones_like(h)], dim=1))
    elif order == 4:
        y_2 = y * y
        x_2 = x * x
        x_3 = x_2 * x
        y_3 = y_2 * y
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2,
                       torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        h_4 = h_3 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, h_4, h_4, h_4, h_4, h_4, torch.ones_like(h)], dim=1))
    else:
        raise ValueError("Polynomial order unsupported, please use 1 or 2 ")

    XtX = torch.matmul(A.permute(0, 2, 1),  w_vector * A)
    XtY = torch.matmul(A.permute(0, 2, 1), w_vector * z)

    beta = solve_linear_system(XtX, XtY, sub_batch_size=16)

    if order > 1: #remove preconditioning
         beta = torch.matmul(D_inv, beta)

    n_est = torch.nn.functional.normalize(torch.cat([-beta[:, 0:2].squeeze(-1), torch.ones(batch_size, 1, device=x.device, dtype=beta.dtype)], dim=1), p=2, dim=1)

    if compute_neighbor_normals:
        beta_ = beta.squeeze().unsqueeze(1).repeat(1, n_points, 1).unsqueeze(-1)
        if order == 1:
            neighbor_normals = n_est.unsqueeze(1).repeat(1, n_points, 1)
        elif order == 2:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)
        elif order == 3:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y + 3 * beta_[:, :, 5] *  x_2 +
                             2 *beta_[:, :, 7] * xy + beta_[:, :, 8] * y_2),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x + 3 * beta_[:, :, 6] * y_2 +
                             beta_[:, :, 7] * x_2 + 2 * beta_[:, :, 8] * xy),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)
        elif order == 4:
            # [x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y + 3 * beta_[:, :, 5] * x_2 +
                             2 * beta_[:, :, 7] * xy + beta_[:, :, 8] * y_2 + 4 * beta_[:, :, 9] * x_3 + 3 * beta_[:, :, 11] * x_2 * y
                             + beta_[:, :, 12] * y_3 + 2 * beta_[:, :, 13] * y_2 * x),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x + 3 * beta_[:, :, 6] * y_2 +
                             beta_[:, :, 7] * x_2 + 2 * beta_[:, :, 8] * xy + 4 * beta_[:, :, 10] * y_3 + beta_[:, :, 11] * x_3 +
                             3 * beta_[:, :, 12] * x * y_2 + 2 * beta_[:, :, 13] * y * x_2),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)

    return beta.squeeze(), n_est, neighbor_normals


def solve_linear_system(XtX, XtY, sub_batch_size=None):
    """
    Solve linear system of equations. use sub batches to avoid MAGMA bug
    :param XtX: matrix of the coefficients
    :param XtY: vector of the
    :param sub_batch_size: size of mini mini batch to avoid MAGMA error, if None - uses the entire batch
    :return:
    """
    if sub_batch_size is None:
        sub_batch_size = XtX.size(0)
    n_iterations = int(XtX.size(0) / sub_batch_size)
    assert sub_batch_size%sub_batch_size == 0, "batch size should be a factor of {}".format(sub_batch_size)
    beta = torch.zeros_like(XtY)
    n_elements = XtX.shape[2]
    for i in range(n_iterations):
        try:
            L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
            beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
        except:
            # # add noise to diagonal for cases where XtX is low rank
            eps = torch.normal(torch.zeros(sub_batch_size, n_elements, device=XtX.device),
                               0.01 * torch.ones(sub_batch_size, n_elements, device=XtX.device))
            eps = torch.diag_embed(torch.abs(eps))
            XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] + \
                eps * XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...]
            try:
                L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                    torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
            except:
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...], _ =\
                    torch.solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...])
    return beta


def compute_principal_curvatures(beta):
    """
    given the jet coefficients, compute the principal curvatures and principal directions:
    the eigenvalues and eigenvectors of the weingarten matrix
    :param beta: batch of Jet coefficients vector
    :return: k1, k2, dir1, dir2: batch of principal curvatures and principal directions
    """
    with torch.no_grad():
        if beta.shape[1] < 5:
            raise ValueError("Can't compute curvatures for jet with order less than 2")
        else:
            b1_2 = torch.pow(beta[:, 0], 2)
            b2_2 = torch.pow(beta[:, 1], 2)
            #first fundemental form
            E = (1 + b1_2).view(-1, 1, 1)
            G = (1 + b2_2).view(-1, 1, 1)
            F = (beta[:, 1] * beta[:, 0]).view(-1, 1, 1)
            I = torch.cat([torch.cat([E, F], dim=2), torch.cat([F, G], dim=2)], dim=1)
            # second fundemental form
            norm_N0 = torch.sqrt(b1_2 + b2_2 + 1)
            e = (2*beta[:, 2] / norm_N0).view(-1, 1, 1)
            f = (beta[:, 4] / norm_N0).view(-1, 1, 1)
            g = (2*beta[:, 3] / norm_N0).view(-1, 1, 1)
            II = torch.cat([torch.cat([e, f], dim=2), torch.cat([f, g], dim=2)], dim=1)

            M_weingarten = -torch.bmm(torch.inverse(I), II)

            curvatures, dirs = torch.symeig(M_weingarten, eigenvectors=True) #faster
            dirs = torch.cat([dirs, torch.zeros(dirs.shape[0], 2, 1, device=dirs.device)], dim=2) # pad zero in the normal direction

    return curvatures, dirs
