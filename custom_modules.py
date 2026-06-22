import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

# ==========================================
# 1. C2f_RFAConv 模組群
# ==========================================
class RFAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(RFAConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        padding = kernel_size // 2
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.c2 = out_channels  # [關鍵修正] 向 YOLOv8 主動宣告輸出通道

    def forward(self, x):
        b, c, h, w = x.size()
        attn = self.attention(x).view(b, 1, self.kernel_size, self.kernel_size)
        out = self.bn(self.conv(x))
        scale = F.adaptive_avg_pool2d(out, 1).sigmoid()
        return self.act(out * scale)

class Bottleneck_RFAConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = RFAConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
        self.c2 = c2  # [關鍵修正] 主動宣告輸出通道

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_RFAConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_RFAConv(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
        self.c2 = c2  # [關鍵修正] 防止系統瞎猜通道數，維持網路結構對齊

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ==========================================
# 2. GSConv & VoV_GSCSP 模組群
# ==========================================
class GSConv(nn.Module):
    """ Slim-Neck 核心：輕量化混合卷積 """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None: 
            p = k // 2
            
        # 第 1 步：標準卷積，這裡負責處理可能的下採樣 (接收外部傳入的 stride = s)
        self.cv1 = Conv(c1, c2 // 2, k, s, p, g=g, act=act)
        
        # [關鍵修正] 第 2 步：深度可分離卷積
        # 這裡的任務是特徵混合，嚴格禁止再次下採樣！
        # 強制固定參數：kernel_size=5, stride=1, padding=2
        self.cv2 = nn.Conv2d(c2 // 2, c2 // 2, 5, 1, 2, groups=c2 // 2, bias=False)
        
        self.bn = nn.BatchNorm2d(c2 // 2)
        self.act = nn.SiLU() if act else nn.Identity()
        self.c2 = c2

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.act(self.bn(self.cv2(x1)))
        
        # 此時 x1 和 x2 的長寬 (H, W) 將完美一致，可順利進行拼接
        return torch.cat((x1, x2), dim=1)

class VoV_GSCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(GSConv(c_, c_, 3, 1) for _ in range(n)))
        self.c2 = c2  # [關鍵修正] 主動宣告輸出通道

    def forward(self, x):
        x1 = self.m(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), dim=1))


# ==========================================
# 3. Four-Detect-ASFF (終極防護版)
# ==========================================
class ASFF_Detect(Detect):
    def __init__(self, *args, **kwargs):
        # 1. 預設值
        nc = kwargs.get('nc', 80)
        ch = kwargs.get('ch', ())
        
        # 2. 嚴謹的型態過濾器：只抓取整數 (nc) 與整數陣列 (ch)，無視系統塞入的其他 None 雜訊
        for arg in args:
            if isinstance(arg, int) and not isinstance(arg, bool):
                nc = arg
            elif isinstance(arg, (list, tuple)) and len(arg) > 0 and all(isinstance(x, int) for x in arg):
                ch = arg

        # 3. 安全初始化底層檢測頭
        super().__init__(nc=nc, ch=ch)
        
        # 4. 構建自適應特徵融合結構
        self.nl = len(ch)
        self.align_convs = nn.ModuleList()
        for i in range(self.nl):
            layer_convs = nn.ModuleList()
            for j in range(self.nl):
                layer_convs.append(nn.Conv2d(ch[j], ch[i], 1, 1, 0) if i != j else nn.Identity())
            self.align_convs.append(layer_convs)
            
        self.asff_weights = nn.Parameter(torch.ones(self.nl, self.nl))

    def forward(self, x):
        weights = torch.softmax(self.asff_weights, dim=1)
        fused_x = []
        for i in range(self.nl):
            target_shape = x[i].shape[2:]
            feat = self.align_convs[i][i](x[i]) * weights[i, i]
            for j in range(self.nl):
                if i != j:
                    aligned = self.align_convs[i][j](x[j])
                    aligned = F.interpolate(aligned, size=target_shape, mode='nearest')
                    feat = feat + aligned * weights[i, j]
            fused_x.append(feat)
        return super().forward(fused_x)
