import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils & Basic Blocks
# -------------------------

def _is_int_sqrt(n: int):
    r = int(math.isqrt(n))
    return r, r * r == n

class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt-v2 风格), 适配 NCHW
    y = x * (gamma * (||x|| / mean(||x||)) + beta) + x
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # L2 范数 over H,W，再做跨通道归一
        gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)  # (B, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：大核 DWConv + 1x1 PWConv
    """
    def __init__(self, channels, kernel_size=13, act=True, norm=True):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels) if norm else nn.Identity()
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class StochasticDepth(nn.Module):
    """
    DropPath / StochasticDepth (按 batch 维度)
    """
    def __init__(self, p, mode="row"):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div(keep)
        return x * mask


class Residual(nn.Module):
    def __init__(self, module, drop_path=0.0):
        super().__init__()
        self.m = module
        self.drop = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        return x + self.drop(self.m(x))


# -------------------------
# Selective Kernel Gate (改造 ScaleAwareGate)
# -------------------------

class SKConv(nn.Module):
    """
    Selective Kernel：多个分支不同感受野，自适应选择融合
    """
    def __init__(self, channels, reduction=8, branches=(('conv',1,0), ('conv',3,1), ('dilated',3,2))):
        """
        branches: list of (type, k, dilation_or_padding)
          ('conv', k, pad) => 普通卷积
          ('dilated', k, dilation) => 空洞卷积
        """
        super().__init__()
        self.branches = nn.ModuleList()
        for btype, k, pd in branches:
            if btype == 'conv':
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, k, padding=pd, groups=channels, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.Hardswish()
                    )
                )
            elif btype == 'dilated':
                pad = pd
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, k, padding=pad, dilation=pad, groups=channels, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.Hardswish()
                    )
                )
            else:
                raise ValueError("Unknown branch type")

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(channels // reduction, len(self.branches) * channels, 1, bias=True),
        )
        self.channels = channels
        self.n_branches = len(self.branches)

    def forward(self, x):
        feats = [b(x) for b in self.branches]  # 每个 (B,C,H,W)
        U = sum(feats)
        s = self.fc(U)  # (B, n_branches*C, 1, 1)
        a = s.view(x.size(0), self.n_branches, self.channels, 1, 1)
        a = torch.softmax(a, dim=1)  # 对分支维度 softmax
        out = 0
        for i in range(self.n_branches):
            out = out + a[:, i] * feats[i]
        return out


class SKScaleAwareGate(nn.Module):
    """
    替换原 ScaleAwareGate：
      - local 分支：SKConv 多感受野选择
      - global 分支：1x1 conv + BN，尺寸对齐后相加
      - 门控：global_act => Hardsigmoid，调制 local 分支
    """
    def __init__(self, inp, oup, sk_reduction=8):
        super().__init__()
        self.local_embed = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        self.local_bn = nn.BatchNorm2d(oup)
        self.local_sk = SKConv(oup, reduction=sk_reduction)

        self.global_embedding = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        self.global_bn = nn.BatchNorm2d(oup)

        self.global_act = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        self.global_act_bn = nn.BatchNorm2d(oup)
        self.act = nn.Hardsigmoid()

    def forward(self, x_l, x_g):
        Bl, Cl, Hl, Wl = x_l.shape

        local = self.local_embed(x_l)
        local = self.local_bn(local)
        local = self.local_sk(local)  # 多尺度选择增强

        global_feat = self.global_embedding(x_g)
        global_feat = self.global_bn(global_feat)
        global_feat = F.interpolate(global_feat, size=(Hl, Wl), mode='bilinear', align_corners=False)

        gact = self.global_act(x_g)
        gact = self.global_act_bn(gact)
        gact = self.act(gact)
        gact = F.interpolate(gact, size=(Hl, Wl), mode='bilinear', align_corners=False)

        out = local * gact + global_feat
        return out


class AblationScaleFusion(nn.Module):
    """
    用于 SKSAG 消融实验的融合模块。

    mode='add':
        local + global_feat
        不使用 gate，不使用 concat，输出 shape 与 local 保持一致。

    mode='concat':
        concat(local, global_feat) -> 1×1 conv
        不使用 gate，通过 1×1 conv 将 2C 压回 C，输出 shape 与 local 保持一致。

    为了公平比较，这里保留 local_embed 和 global_embedding，
    使 local 和 global_feat 在通道数和空间尺寸上对齐。
    """
    def __init__(self, inp, oup, mode="add", use_local_sk=True, sk_reduction=8):
        super().__init__()
        assert mode in ["add", "concat"], "mode must be 'add' or 'concat'"
        self.mode = mode

        # local 分支：将 x_l 映射到 oup 通道
        self.local_embed = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        self.local_bn = nn.BatchNorm2d(oup)

        # 为了只消融 gate，也可以保留 SKConv；
        # 如果你想做更纯粹的 add/concat，可以设置 use_local_sk=False。
        self.use_local_sk = use_local_sk
        self.local_sk = SKConv(oup, reduction=sk_reduction) if use_local_sk else nn.Identity()

        # global 分支：将 x_g 映射到 oup 通道，并上采样到 local 尺寸
        self.global_embedding = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        self.global_bn = nn.BatchNorm2d(oup)

        # concat 消融时，需要把 2C 压回 C，保证输出 shape 不变
        if self.mode == "concat":
            self.concat_fuse = nn.Sequential(
                nn.Conv2d(oup * 2, oup, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardswish()
            )
        else:
            self.concat_fuse = nn.Identity()

    def forward(self, x_l, x_g):
        _, _, Hl, Wl = x_l.shape

        # local feature
        local = self.local_embed(x_l)
        local = self.local_bn(local)
        local = self.local_sk(local)

        # global feature
        global_feat = self.global_embedding(x_g)
        global_feat = self.global_bn(global_feat)
        global_feat = F.interpolate(
            global_feat,
            size=(Hl, Wl),
            mode='bilinear',
            align_corners=False
        )

        if self.mode == "add":
            # 消融 1：直接相加
            out = local + global_feat

        elif self.mode == "concat":
            # 消融 2：concat 后 1×1 conv 压回原通道
            out = torch.cat([local, global_feat], dim=1)
            out = self.concat_fuse(out)

        return out

# -------------------------
# BiFPN (单层轻量实现)
# -------------------------

class BiFPNBlock(nn.Module):
    """
    4 层金字塔的轻量 BiFPN（单次 top-down + bottom-up），
    现在加入通道适配器，确保相加前通道一致。
    """
    def __init__(self, channels):
        super().__init__()
        assert len(channels) == 4
        self.channels = channels
        C1, C2, C3, C4 = channels

        # 上采样/下采样
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2)
        self.eps = 1e-6

        # —— 顶向下 Top-Down —— #
        # 权重
        self.w3_td = nn.Parameter(torch.ones(2))  # P3 + up(P4)
        self.w2_td = nn.Parameter(torch.ones(2))  # P2 + up(P3_td)
        self.w1_td = nn.Parameter(torch.ones(2))  # P1 + up(P2_td)
        # 通道适配器：把上一级的上采样结果投到当前层通道
        self.adapt_p4_to_p3 = nn.Conv2d(C4, C3, kernel_size=1, bias=False)
        self.adapt_p3_to_p2 = nn.Conv2d(C3, C2, kernel_size=1, bias=False)
        self.adapt_p2_to_p1 = nn.Conv2d(C2, C1, kernel_size=1, bias=False)
        # 融合后的深度可分离卷积
        def dsconv(c):  # depthwise separable conv + BN + act
            return DepthwiseSeparableConv(c, kernel_size=5)
        self.conv3_td = dsconv(C3)
        self.conv2_td = dsconv(C2)
        self.conv1_td = dsconv(C1)

        # —— 自底向上 Bottom-Up —— #
        self.w2_bu = nn.Parameter(torch.ones(2))  # P2_td + down(P1_td)
        self.w3_bu = nn.Parameter(torch.ones(3))  # P3_td + up(P4_td) + down(P2_out)
        self.w4_bu = nn.Parameter(torch.ones(2))  # P4_td + down(P3_out)
        # 通道适配器：把来自其它层的特征投到目标通道
        self.adapt_p1_to_p2 = nn.Conv2d(C1, C2, kernel_size=1, bias=False)
        self.adapt_p4_to_p3_bu = nn.Conv2d(C4, C3, kernel_size=1, bias=False)
        self.adapt_p2_to_p3 = nn.Conv2d(C2, C3, kernel_size=1, bias=False)
        self.adapt_p3_to_p4 = nn.Conv2d(C3, C4, kernel_size=1, bias=False)

        self.conv2_bu = dsconv(C2)
        self.conv3_bu = dsconv(C3)
        self.conv4_bu = dsconv(C4)

    def _norm_w(self, w):
        w = F.relu(w)
        return w / (w.sum() + self.eps)

    def forward(self, inputs):
        # inputs: [P1,P2,P3,P4]（高分→低分）
        P1, P2, P3, P4 = inputs

        # —— Top-Down —— #
        P4_td = P4
        w = self._norm_w(self.w3_td)
        P3_up = self.upsample(P4_td)
        P3_up = self.adapt_p4_to_p3(P3_up)                 # 通道对齐到 C3
        P3_td = self.conv3_td(w[0] * P3 + w[1] * P3_up)

        w = self._norm_w(self.w2_td)
        P2_up = self.upsample(P3_td)
        P2_up = self.adapt_p3_to_p2(P2_up)                 # 通道对齐到 C2
        P2_td = self.conv2_td(w[0] * P2 + w[1] * P2_up)

        w = self._norm_w(self.w1_td)
        P1_up = self.upsample(P2_td)
        P1_up = self.adapt_p2_to_p1(P1_up)                 # 通道对齐到 C1
        P1_td = self.conv1_td(w[0] * P1 + w[1] * P1_up)

        # —— Bottom-Up —— #
        w = self._norm_w(self.w2_bu)
        P1_down = self.downsample(P1_td)
        P1_down = self.adapt_p1_to_p2(P1_down)             # 通道对齐到 C2
        P2_out = self.conv2_bu(w[0] * P2_td + w[1] * P1_down)

        w = self._norm_w(self.w3_bu)
        P4_up = self.upsample(P4_td)
        P4_up = self.adapt_p4_to_p3_bu(P4_up)              # 通道对齐到 C3
        P2_down = self.downsample(P2_out)
        P2_down = self.adapt_p2_to_p3(P2_down)             # 通道对齐到 C3
        P3_out = self.conv3_bu(w[0] * P3_td + w[1] * P4_up + w[2] * P2_down)

        w = self._norm_w(self.w4_bu)
        P3_down = self.downsample(P3_out)
        P3_down = self.adapt_p3_to_p4(P3_down)             # 通道对齐到 C4
        P4_out = self.conv4_bu(w[0] * P4_td + w[1] * P3_down)

        return [P1_td, P2_out, P3_out, P4_out]



class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNBlock(channels) for _ in range(num_layers)])

    def forward(self, xs):
        for l in self.layers:
            xs = l(xs)
        return xs


# -------------------------
# Attention (修复 & 并行大核卷积分支)
# -------------------------

class Attention(nn.Module):
    """
    简化与修复版：
      - 输入 x_seq:(B,N,C)，内部做多头注意力
      - 同时对“最高分辨率前 n0 个 token”走一个大核 DWConv 卷积分支（空间增强）
      - 输出只返回前 n0 个 token，经线性映射回 (B, C, H, W)
    """
    def __init__(self, dim, num_heads=8, head_dim=32, dw_kernel=13):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.proj = nn.Linear(self.inner_dim, dim, bias=False)

        # 并行的卷积分支（作用于最高分辨率子序列）
        self.conv_enhance = nn.Sequential(
            DepthwiseSeparableConv(self.inner_dim, kernel_size=dw_kernel),
            nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Hardswish()
        )

    def forward(self, x_seq, img_hw):
        """
        x_seq: (B, N, C)  [N = n0 + n1 + n2]
        img_hw: (H, W)    [最高分辨率 n0 = H*W]
        return: (B, C, H, W)
        """
        B, N, C = x_seq.shape
        H0, W0 = img_hw
        n0 = H0 * W0
        assert N >= n0, "N must be >= H0*W0"
        qkv = self.qkv(x_seq)  # (B,N,3*inner)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, n_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.x: 使用 SDPA（自动做缩放和 softmax）
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, nH, N, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.inner_dim)  # (B,N,inner)
        attn_out = self.proj(attn_out)  # (B,N,dim)

        # 只保留最高分辨率的前 n0 个 token
        x0 = attn_out[:, :n0, :]  # (B,n0,dim)
        x0 = x0.transpose(1, 2).contiguous().view(B, C, H0, W0)  # (B,dim,H0,W0)

        # 并行大核卷积分支：使用 v 的前 n0 个 token（更贴近注意力值）
        v0 = v[:, :, :n0, :]  # (B,nH,n0,head_dim)
        v0 = v0.transpose(1, 2).contiguous().view(B, n0, self.inner_dim)  # (B,n0,inner)
        v0 = v0.transpose(1, 2).contiguous().view(B, self.inner_dim, H0, W0)
        v_enh = self.conv_enhance(v0)  # (B,dim,H0,W0)

        return x0 + v_enh


# -------------------------
# Cross-Scale Attention (修复版)
# -------------------------

class CrossScaleAttention(nn.Module):
    """
    输入 x:(B,C,H,W)
      - 生成三尺度序列：x0=(H,W), x1≈(H/2,W/2), x2≈(H/3,W/3)
      - 拼接为 (B,N,C) 输入 Attention
      - 输出回 (B,C,H,W)
    """
    def __init__(self, dim, dw_kernel=13, num_heads=8, head_dim=32):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

        # 下采样支路（可替代为自适应池化）
        self.dw1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=3, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.attn = Attention(dim, num_heads=num_heads, head_dim=head_dim, dw_kernel=dw_kernel)
        self.out_norm = nn.BatchNorm2d(dim)
        self.out_act = nn.Hardswish()
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x0 = self.norm(x)
        x1 = self.dw1(x0)
        x2 = self.dw2(x0)

        # flatten 到序列
        x0_seq = x0.flatten(2).transpose(1, 2)  # (B,H*W,C)
        x1_seq = x1.flatten(2).transpose(1, 2)
        x2_seq = x2.flatten(2).transpose(1, 2)
        seq = torch.cat([x0_seq, x1_seq, x2_seq], dim=1)  # (B,N,C)

        y0 = self.attn(seq, img_hw=(H, W))  # (B,C,H,W) 只输出最高分辨率子序列
        y0 = self.out_conv(self.out_act(self.out_norm(y0)))
        return y0


# -------------------------
# FeedForward with SwiGLU + GRN + 大核 DWConv
# -------------------------

class FeedForward(nn.Module):
    """
    1x1(2*hidden) -> SwiGLU -> 大核DWConv -> GRN -> 1x1(out=dim)
    """
    def __init__(self, dim, hidden_dim, dw_kernel=13):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, 2*hidden_dim, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=dw_kernel, padding=dw_kernel//2, groups=hidden_dim, bias=False)
        self.grn = GRN(hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.pre_norm(x)
        u, v = self.fc1(x).chunk(2, dim=1)
        x = F.silu(u) * v                   # SwiGLU variant
        x = self.dw(x)
        x = self.grn(x)
        x = self.fc2(x)
        return x


class IntraFeedForward(nn.Module):
    """
    沿用你的切分思路，但更稳妥为等分 4 份（避免与外部 channels 耦合）
    """
    def __init__(self, dim, mlp_ratio=2, dw_kernel=13, drop_path=0.0):
        super().__init__()
        q = dim // 4
        self.splits = [q, q, q, dim - 3*q]
        self.ff1 = Residual(FeedForward(self.splits[0], mlp_ratio*self.splits[0], dw_kernel), drop_path)
        self.ff2 = Residual(FeedForward(self.splits[1], mlp_ratio*self.splits[1], dw_kernel), drop_path)
        self.ff3 = Residual(FeedForward(self.splits[2], mlp_ratio*self.splits[2], dw_kernel), drop_path)
        self.ff4 = Residual(FeedForward(self.splits[3], mlp_ratio*self.splits[3], dw_kernel), drop_path)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, self.splits, dim=1)
        x1 = self.ff1(x1)
        x2 = self.ff2(x2)
        x3 = self.ff3(x3)
        x4 = self.ff4(x4)
        return torch.cat([x1, x2, x3, x4], dim=1)


# -------------------------
# CIM Block（集成 CSA + IntraFF + CSA + FF）
# -------------------------

class CIMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2, dw_kernel=13, num_heads=8, head_dim=32, drop_path=0.0):
        super().__init__()
        self.csa1 = Residual(CrossScaleAttention(dim, dw_kernel, num_heads, head_dim), drop_path)
        self.intra_ff = Residual(IntraFeedForward(dim, mlp_ratio, dw_kernel, drop_path), drop_path)
        self.csa2 = Residual(CrossScaleAttention(dim, dw_kernel, num_heads, head_dim), drop_path)
        self.ff = Residual(FeedForward(dim, dim*mlp_ratio, dw_kernel), drop_path)

    def forward(self, x):
        x = self.csa1(x)
        x = self.intra_ff(x)
        x = self.csa2(x)
        x = self.ff(x)
        return x


# -------------------------
# PyramidPoolAgg（沿用你的逻辑）
# -------------------------

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        # inputs: list of 4 features, use the last one's H,W 作为基准下采样
        B, C, H, W = inputs[-1].shape
        Ht = (H - 1) // self.stride + 1
        Wt = (W - 1) // self.stride + 1
        pooled = [F.adaptive_avg_pool2d(inp, (Ht, Wt)) for inp in inputs]
        return torch.cat(pooled, dim=1)  # (B, sum(Ci), Ht, Wt)


# -------------------------
# CIM（重构版）
# -------------------------

class CIM(nn.Module):
    """
    输入：tuple/list of 4 tensors: [B,C1,H1,W1], [B,C2,H2,W2], [B,C3,H3,W3], [B,C4,H4,W4]
    输出：同尺寸的 4 个张量列表

    fusion_mode:
        'sksag'  : 原始 SKSAG，local * gate + global_feat
        'add'    : 消融，local + global_feat
        'concat' : 消融，concat(local, global_feat) -> 1×1 conv
    """
    def __init__(
        self,
        dim,
        num_layers=1,
        channels=[128,256,512,1024],
        downsample=1,
        bifpn_layers=1,
        mlp_ratio=2,
        dw_kernel=13,
        num_heads=8,
        head_dim=32,
        drop_path=0.0,
        fusion_mode="sksag",
        use_local_sk_in_ablation=True
    ):
        super().__init__()
        assert sum(channels) == dim, "dim should equal sum(channels)"
        assert fusion_mode in ["sksag", "add", "concat"], \
            "fusion_mode must be 'sksag', 'add', or 'concat'"

        self.channels = channels
        self.hidden_dim = dim // 4
        self.stride = downsample
        self.fusion_mode = fusion_mode

        # 先做 BiFPN 融合，再进入 CIM 主干
        self.bifpn = BiFPN(channels, num_layers=bifpn_layers)

        self.pool = PyramidPoolAgg(stride=self.stride)
        self.down_channel = nn.Conv2d(dim, self.hidden_dim, kernel_size=1, bias=False)

        self.blocks = nn.Sequential(*[
            CIMBlock(self.hidden_dim, mlp_ratio, dw_kernel, num_heads, head_dim, drop_path)
            for _ in range(num_layers)
        ])
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.up_channel = nn.Conv2d(self.hidden_dim, dim, kernel_size=1, bias=False)

        # 三种融合方式：原始 SKSAG / add 消融 / concat 消融
        if fusion_mode == "sksag":
            self.fusion = nn.ModuleList([
                SKScaleAwareGate(channels[i], channels[i])
                for i in range(4)
            ])
        else:
            self.fusion = nn.ModuleList([
                AblationScaleFusion(
                    channels[i],
                    channels[i],
                    mode=fusion_mode,
                    use_local_sk=use_local_sk_in_ablation
                )
                for i in range(4)
            ])

    def forward(self, inputs):
        # 1) BiFPN 融合
        feats = self.bifpn(list(inputs))  # list of 4

        # 2) Pool -> CIM 主干 (统一分辨率处理)
        out = self.pool(feats)           # (B, sumC, Ht, Wt)
        out = self.down_channel(out)     # (B, hidden, Ht, Wt)
        out = self.blocks(out)
        out = self.bn(out)
        out = self.up_channel(out)       # (B, sumC, Ht, Wt)

        # 3) 按通道切分 -> 与各层做 SKScaleAwareGate 融合（global -> upsample 到各自分辨率）
        xs_global = out.split(self.channels, dim=1)
        results = []
        for i in range(4):
            local = feats[i]         # BiFPN 后的局部特征（与输入同分辨率）
            global_i = xs_global[i]  # CIM 全局上下文的该层通道片段（Ht,Wt）
            fused = self.fusion[i](local, global_i)
            results.append(fused)
        return results


# -------------------------
# Tests
# -------------------------

def _print_shapes(ys):
    for i, y in enumerate(ys):
        print(f"y[{i}]:", tuple(y.shape))

def _sanity_check():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 构造 4 层输入（和你原例一致）
    B = 2
    x1 = torch.randn(B, 128, 128, 128).to(device)
    x2 = torch.randn(B, 256, 64, 64).to(device)
    x3 = torch.randn(B, 512, 32, 32).to(device)
    x4 = torch.randn(B, 1024, 16, 16).to(device)
    x = (x1, x2, x3, x4)

    model = CIM(
        dim=1920,
        num_layers=1,
        channels=[128,256,512,1024],
        downsample=2,      # 统一分辨率（可调）
        bifpn_layers=1,
        mlp_ratio=2,
        dw_kernel=13,
        num_heads=8,
        head_dim=32,
        drop_path=0.0
    ).to(device)

    model.eval()
    with torch.no_grad():
        ys = model(x)
        _print_shapes(ys)   # 期望分别为 (B,128,128,128), (B,256,64,64), (B,512,32,32), (B,1024,16,16)

    # 简单的前向耗时（可选）
    model.train()
    iters = 5
    import time
    t0 = time.time()
    for _ in range(iters):
        ys = model(x)
        loss = sum(y.mean() for y in ys)
        loss.backward()
    t1 = time.time()
    print(f"{iters} iters train-mode forward+backward time: {t1-t0:.3f}s")


if __name__ == "__main__":
    _sanity_check()
