import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSCM(nn.Module):
    """结构因果模型(SCM)模块，用于分离类别相关和类别无关特征"""

    def __init__(self, feature_dim, z_dim=64, class_dim=32, style_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.class_dim = class_dim
        self.style_dim = style_dim

        # 编码器：将特征分解为类别相关(c)和类别无关(s)部分
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, z_dim)
        )

        # 类别内容编码器 (C) - 提取类别本质特征
        self.content_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, class_dim)
        )

        # 风格编码器 (S) - 提取类别无关特征(背景、光照等)
        self.style_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, style_dim)
        )

        # 解码器：从c和s重建特征
        self.decoder = nn.Sequential(
            nn.Linear(class_dim + style_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, feature_dim)
        )

        # 用于结构残差干预的MLP
        self.intervention_mlp = nn.Sequential(
            nn.Linear(class_dim, class_dim),
            nn.ReLU(),
            nn.Linear(class_dim, class_dim)
        )

        # 添加类别特征的投影器，将类别特征投影回原始特征空间
        self.class_projector = nn.Sequential(
            nn.Linear(class_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def encode(self, x):
        """将特征编码为潜在表示"""
        z = self.encoder(x)
        c = self.content_encoder(z)  # 类别内容
        s = self.style_encoder(z)  # 风格内容
        return c, s

    def decode(self, c, s):
        """从内容和风格解码回特征"""
        z_combined = torch.cat([c, s], dim=1)
        return self.decoder(z_combined)

    def intervene(self, c, strength=1.0):
        """对类别内容特征进行干预，强化类别信息"""
        c_intervened = c + strength * self.intervention_mlp(c)
        return c_intervened

    def counterfactual(self, x, y, alpha=0.8):
        """增强的反事实生成：针对非判别性特征进行干预

        Args:
            x: 原始特征
            y: 标签
            alpha: 混合强度参数
        """
        # 编码获取类别特征和风格特征
        c_x, s_x = self.encode(x)

        # 对批次内样本的风格特征进行干预（非判别性特征）
        batch_size = x.size(0)

        # 为每个样本找到不同类别的样本
        perm_idx = torch.randperm(batch_size)

        # 获取不同样本的风格特征
        s_other = s_x[perm_idx]

        # 混合原始风格特征和其他样本的风格特征
        s_mixed = alpha * s_x + (1 - alpha) * s_other

        # 使用原始类别特征和混合风格特征生成反事实样本
        x_cf = self.decode(c_x, s_mixed)

        return x_cf, c_x, s_mixed

    def forward(self, x, intervention_strength=0.5, training=True):
        """前向传播，包含编码-干预-解码过程"""
        # 编码
        c, s = self.encode(x)

        # 对类别内容进行干预
        if training:
            c = self.intervene(c, intervention_strength)

        # 重建特征
        x_recon = self.decode(c, s)

        # 将类别特征投影到原始特征空间
        c_projected = self.class_projector(c)

        # 返回干预后的类别特征、风格特征和重建特征
        return {
            'class_features': c,
            'class_features_projected': c_projected,
            'style_features': s,
            'reconstructed_features': x_recon,
            'original_features': x
        }