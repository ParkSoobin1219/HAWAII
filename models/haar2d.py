import torch
import torch.nn.functional as F


def Haar(img, device=None):
    if len(img.shape) == 4:
        B,C,H,W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    else:
        img = img.unsqueeze(dim=0)
        B,C,H,W = img.shape[0],img.shape[1], img.shape[2], img.shape[3]
    # Haar filter with size of (4, 1, 2, 2)
    haar_filters_single = torch.tensor([
        [[1/4, 1/4], [1/4, 1/4]],  # LL
        [[-1, -1], [1, 1]],  # LH
        [[-1, 1], [-1, 1]],  # HL
        [[1, -1], [-1, 1]]  # HH
    ], dtype=torch.float32).view(4, 1, 2, 2).to(device)

    # duplicate single filter to C channel (4*C, 1, 2, 2)
    haar_filters = haar_filters_single.repeat(C, 1, 1, 1).reshape(4*C, 1, 2, 2).to(device)

    # (C channel, 4 filter -> C*4 result)
    output = F.conv2d(img.view(B, C*1, H, W), haar_filters, stride=2, groups=C).to(device)

    # separate in ll,lh,hl,hh (B, C, 4, H, W)
    output = output.view(B, C, 4, H//2, W//2).to(device)

    return output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3] # ll, lh, hl, hh
