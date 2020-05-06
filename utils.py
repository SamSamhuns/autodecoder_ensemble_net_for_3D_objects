import torch
from torch.autograd import Variable

def chamfer_loss(x, y, ps=91):
    A = x.cuda()
    B = y.cuda()
    A = A.permute(0, 2, 1)
    B = B.permute(0, 2, 1)
    r = torch.sum(A * A, dim=2)
    r = r.unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2)
    r1 = r1.unsqueeze(-1)
    t = (r.repeat(1, 1, ps) - 2 * torch.bmm(A, B.permute(0, 2, 1)) +
         r1.permute(0, 2, 1).repeat(1, ps, 1))
    d1, _ = t.min(dim=1)
    d2, _ = t.min(dim=2)
    ls = (d1 + d2) / 2
    return ls.mean()
