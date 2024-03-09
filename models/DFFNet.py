import torch
import torch.nn as nn
import torch.nn.functional as F


class SFFM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv3_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.conv5 = nn.Conv2d(2 * dim, 4 * dim, 1)
        self.conv55 = nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim)
        self.conv6 = nn.Conv2d(4 * dim, dim, 1)

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.gelu(x)

        # PA
        x2 = self.pa(x) * x

        # LKA
        attn = self.conv2(x)
        attn = self.conv3_spatial(attn)
        attn = self.conv4(attn)
        attn = self.sigmoid(attn)
        x3 = x * attn

        x = self.gelu(self.conv55(self.conv5(torch.cat([x2, x3], dim=1))))
        x = self.conv6(x)
        x = identity + x

        identity = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [SFFM(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class FFFM(nn.Module):
    def __init__(self, features, M=3, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features
        self.convs = nn.ModuleList([])

        self.convh = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.convm = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fc1 = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.fcs1 = nn.ModuleList([])
        for i in range(M):
            self.fcs1.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        self.out1 = nn.Conv2d(features, features, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros((1, features, 1, 1)), requires_grad=True)

    def forward(self, x):
        low = self.convl(x)
        middle = self.convm(low)
        high = self.convh(middle)
        emerge = low + middle + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        middle_att = self.fcs[1](fea_z)
        low_att = self.fcs[2](fea_z)

        attention_vectors = torch.cat([high_att, middle_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, middle_att, low_att = torch.chunk(attention_vectors, 3, dim=1)

        fea_high = high * high_att
        fea_middle = middle * middle_att
        fea_low = low * low_att
        out = self.out(fea_high + fea_middle + fea_low)

        emerge1 = self.gap(out)
        fea_z1 = self.fc1(emerge1)
        high_att1 = self.fcs1[0](fea_z1)
        middle_att1 = self.fcs1[1](fea_z1)
        low_att1 = self.fcs1[2](fea_z1)
        attention_vectors1 = torch.cat([high_att1, middle_att1, low_att1], dim=1)
        attention_vectors1 = self.softmax(attention_vectors1)

        high_att1, middle_att1, low_att1 = torch.chunk(attention_vectors1, 3, dim=1)
        fea_high1 = high * high_att1
        fea_middle1 = middle * middle_att1
        fea_low1 = low * low_att1
        out_a = self.out1(fea_high1 + fea_middle1 + fea_low1)

        return out_a * self.gamma + x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class DFFNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(DFFNet, self).__init__()

        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2] // 2)
        self.sfconv = FFFM(embed_dims[2])
        self.layer4 = BasicLayer(dim=embed_dims[2], depth=depths[2] // 2)

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer5 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer6 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.sfconv(x)
        x = self.layer4(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer5(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer6(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def DFFNet_T():
    return DFFNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])


def DFFNet_S():
    return DFFNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])


def DFFNet_L():
    return DFFNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[8, 8, 16, 8, 8])


from thop import clever_format, profile

if __name__ == '__main__':
    model = DFFNet_S()
    input_data = torch.randn(1, 3, 256, 256)

    # calculate flops and params
    flops, params = profile(model, inputs=(input_data,))
    flops, params = clever_format([flops, params], "%.3f")
    print(params)
    print(flops)
