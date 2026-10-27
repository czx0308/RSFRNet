import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
import timm
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)

class F_FiLM(nn.Module):
    def __init__(self, in_channels, condition_dim):
        super().__init__()
        self.fc = nn.Linear(condition_dim, 4 * in_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x, condition_vector):
        b, c, h, w = x.shape
        params = self.fc(condition_vector)
        g_h, g_l, b_h, b_l = torch.split(params, c, dim=1)

        x_low = self.avg_pool(x)
        x_high = x - x_low

        g_h = g_h.view(b, c, 1, 1)
        g_l = g_l.view(b, c, 1, 1)
        b_h = b_h.view(b, c, 1, 1)
        b_l = b_l.view(b, c, 1, 1)

        out = ((1 + g_h) * x_high + b_h) + ((1 + g_l) * x_low + b_l)
        return out

# class L_Codebook(nn.Module):
#     def __init__(self, in_channels, num_classes=2, num_heads=8):
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#
#         # 定义 Codebook (可学习的类别原型向量)
#         # [Num_Classes, C] -> 0:农田, 1:道路
#         self.class_codebook = nn.Parameter(torch.randn(num_classes, in_channels))
#
#         # 交叉注意力组件
#         self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
#         self.norm_q = nn.LayerNorm(in_channels)
#         self.norm_kv = nn.LayerNorm(in_channels)
#
#         # 融合系数 (初始化为0，确保训练初期不破坏原有特征)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x, class_id):
#         """
#         x: [B, C, H, W] (Encoder输出的e4)
#         class_id: [B] (类别标签)
#         """
#         B, C, H, W = x.shape
#
#         # 1. 准备 Key 和 Value (来自图像特征)
#         # Flatten: [B, C, H, W] -> [B, HW, C]
#         feat_flat = x.flatten(2).transpose(1, 2)
#         k = v = self.norm_kv(feat_flat)
#
#         # 2. 准备 Query (来自 Codebook)
#         # Expand: [Num_Classes, C] -> [B, Num_Classes, C]
#         q = self.class_codebook.unsqueeze(0).expand(B, -1, -1)
#         q = self.norm_q(q)
#
#         # 3. Cross Attention
#         # 这一步计算 Codebook 中每个类别向量在当前图像中的全局特征表示
#         # attn_out: [B, Num_Classes, C]
#         attn_out, _ = self.cross_attn(query=q, key=k, value=v)
#
#         # 4. 根据 class_id 选择对应的增强向量
#         # class_id: [B] -> 需要扩展索引以进行 gather
#         # indices: [B, 1, C]
#         if class_id.dim() == 1:
#             class_id = class_id.unsqueeze(1)  # [B, 1]
#
#         indices = class_id.unsqueeze(-1).expand(-1, -1, C)  # [B, 1, C]
#
#         # 从 attn_out 中取出当前样本对应的类别向量
#         # selected_vec: [B, 1, C]
#         selected_vec = torch.gather(attn_out, 1, indices)
#
#         # 5. 特征增强 (Channel-wise Attention)
#         # 将选出的向量应用回原特征图
#         # selected_vec: [B, 1, C] -> [B, C, 1, 1]
#         modulator = selected_vec.view(B, C, 1, 1)
#
#         # 采用残差形式: 原特征 + gamma * (原特征 * 类别上下文)
#         # 这样模型可以学习强调与该类别相关的通道
#         out = x + self.gamma * (x * modulator)
#
#         return out

class L_Codebook(nn.Module):
    def __init__(self, in_channels, num_classes=2, num_heads=8):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.class_codebook = nn.Parameter(torch.empty(num_classes, in_channels))
        nn.init.orthogonal_(self.class_codebook)
        # 或者 nn.init.normal_(self.class_codebook, std=0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_channels)
        self.norm_kv = nn.LayerNorm(in_channels)

    def forward(self, x, class_id):
        B, C, H, W = x.shape

        feat_flat = x.flatten(2).transpose(1, 2)
        k = v = self.norm_kv(feat_flat)
        q = self.class_codebook.unsqueeze(0).expand(B, -1, -1)
        q = self.norm_q(q)

        attn_out, _ = self.cross_attn(q, k, v)  # [B, Num_Classes, C]

        if class_id.dim() == 1:
            class_id = class_id.unsqueeze(1)
        indices = class_id.unsqueeze(-1).expand(-1, -1, C)
        selected_vec = torch.gather(attn_out, 1, indices)  # [B, 1, C]

        prototype = selected_vec.view(B, C, 1, 1)  # [B, C, 1, 1]

        out = prototype + x

        return out


class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class RSFRNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, backbone_name='resnet50', pretrained=True, condition_dim=32):
        super(RSFRNet, self).__init__()
        self.name = 'RSFRNet'

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        filters = self.backbone.feature_info.channels()

        self.condition_embedding = nn.Embedding(num_embeddings=2, embedding_dim=condition_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim, condition_dim)
        )

        # 3. Attention Layers (PAM/CAM)
        self.attention4 = PAM_CAM_Layer(filters[3])
        self.attention3 = PAM_CAM_Layer(filters[2])
        self.attention2 = PAM_CAM_Layer(filters[1])
        self.attention1 = PAM_CAM_Layer(filters[0])

        self.l_codebook = L_Codebook(in_channels=filters[3], num_classes=2)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.f_film4 = F_FiLM(filters[2], condition_dim)
        self.f_film3 = F_FiLM(filters[1], condition_dim)
        self.f_film2 = F_FiLM(filters[0], condition_dim)
        self.f_film1 = F_FiLM(filters[0], condition_dim)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, class_id):

        c_emb = self.condition_embedding(class_id)
        c_emb = self.condition_mlp(c_emb)  # [Batch, condition_dim]

        e1, e2, e3, e4 = self.backbone(x)

        e4 = self.attention4(e4)
        e4 = self.l_codebook(e4, class_id)


        # Layer 4
        d4 = self.decoder4(e4) + self.attention3(e3)
        d4 = self.f_film4(d4, c_emb)  # F-FiLM 调制

        # Layer 3
        d3 = self.decoder3(d4) + self.attention2(e2)
        d3 = self.f_film3(d3, c_emb)  # F-FiLM 调制

        # Layer 2
        d2 = self.decoder2(d3) + self.attention1(e1)
        d2 = self.f_film2(d2, c_emb)  # F-FiLM 调制

        # Layer 1
        d1 = self.decoder1(d2)
        d1 = self.f_film1(d1, c_emb)  # F-FiLM 调制

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out = torch.sigmoid(out)

        return out


def build_rsfrnet(args=None):
    return RSFRNet(num_classes=1, pretrained=True)


def Multi_MAnetA3(args=None):
    return build_rsfrnet(args)
