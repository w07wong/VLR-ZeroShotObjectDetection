import math
import numpy as np

import torch
import torch.nn.functional as F


# Referece: https://github.com/Zzh-tju/DIoU-pytorch-detectron


def bbox_transform(deltas, weights):
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    #dw = torch.clamp(dw, max=cfg.BBOX_XFORM_CLIP)
    #dh = torch.clamp(dh, max=cfg.BBOX_XFORM_CLIP)

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)


def compute_diou(output, target, transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output).double()
    mask = ((ykis2 > ykis1) * (xkis2 > xkis1))
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u
    iouk = ((1 - iouk) * 1).sum(0) / output.size(0)
    diouk = ((1 - diouk) * 1).sum(0) / output.size(0)

    return iouk, diouk
