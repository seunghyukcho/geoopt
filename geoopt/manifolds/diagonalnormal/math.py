import torch
from math import sqrt


# x shape: (*, N, 2)
def dist(x0, x1, keepdim):
    x0_ = torch.tensor(x0)
    x0_[..., 0] = x0_[..., 0] / sqrt(2)
    x1_ = torch.tensor(x1)
    x1_[..., 0] = x1_[..., 0] / sqrt(2)
    x1__ = torch.tensor(x1_)
    x1__[..., 1] = -x1__[..., 1]

    dist1 = (x0_ - x1__).pow(2).sum(dim=-1).sqrt()  # (*, N)
    dist2 = (x0_ - x1_).pow(2).sum(dim=-1).sqrt()  # (*, N)

    dist = (dist1 + dist2).log() - (dist1 - dist2).log()  # (*, N)
    dist = dist.pow(2).sum(dim=-1, keepdim=keepdim) * 2
    dist = dist.sqrt()

    return dist  # (*, 1)


def riemannian_metric(x):
    # N = x.size()[-2]
    # sz = x.size()[:-2]
    # sz.append(2 * N)
    metric_tensor = torch.empty(x.size(), device=x.device)
    
    metric_tensor[..., 0] = 1 / x[..., 1].pow(2)
    metric_tensor[..., 1] = 2 / x[..., 1].pow(2)

    # metric_tensor[..., torch.arange(0, N, 2)] = 1 / x[..., 1].pow(2)
    # metric_tensor[..., torch.arange(1, N, 2)] = 2 / x[..., 1].pow(2)


    return metric_tensor  # (*, N, 2)


def inv_riemannian_metric(x):
    # N = x.size()[-2]
    # sz = x.size()[:-2]
    # sz.append(2 * N)
    # metric_tensor = torch.empty(sz)

    # metric_tensor[..., torch.arange(0, N, 2)] = x[..., 1].pow(2)
    # metric_tensor[..., torch.arange(1, N, 2)] = x[..., 1].pow(2) / 2
    metric_tensor = torch.empty(x.size(), device=x.device)
    
    metric_tensor[..., 0] = x[..., 1].pow(2)
    metric_tensor[..., 1] = x[..., 1].pow(2) / 2

    return metric_tensor  # (*, N, 2)


# x shape: (*, N, 2), v shape: (*, N, 2)
def exp(x, v):
    # new_x = torch.zeros(v.size())
    # print(v.isnan().sum())
    new_x1 = torch.zeros(v.size(), device=v.device)
    new_x2 = torch.zeros(v.size(), device=v.device)

    metric_tensor = riemannian_metric(x)
    v_norm = v.pow(2) * metric_tensor
    v_norm = v_norm.sum(dim=-1, keepdim=True)

    # v_norm = v.pow(2).sum(dim=-1, keepdim=True)
    r = (v_norm / 2).sqrt()

    x0 = x[..., :1]
    x1 = x[..., 1:]
    v0 = v[..., :1]
    v1 = v[..., 1:]

    sign = torch.where(v1 > 0, 1, -1)
    new_x1[..., :1] = x0
    new_x1[..., 1:] = x1 * (r * sign).exp()

    # print(x0.size(), v1.size())
    p = x0 + 2 * v1 * x1 / (v0 + 1e-9)
    b = ((x0 - p).pow(2) / 2 + x1.pow(2)).sqrt()
    # print(r.min(), v0.min(), b.min())

    # print(p, b)

    t0 = torch.arctanh((x0 - p) / (sqrt(2) * b)) / r
    # print(p.size(), b.size(), r.size(), t0.size())
    sign = torch.where(v0 > 0, 1, -1)
    new_x2[..., :1] = p + sqrt(2) * b * torch.tanh(r * (t0 + sign))
    new_x2[..., 1:] = b / torch.cosh(r * (t0 + sign))

    new_x = torch.where(v0 == 0, new_x1, new_x2)
    # print(new_x.isnan().sum())

    return new_x


if __name__ == "__main__":
    import plotly.graph_objects as go

    x = torch.tensor([[1., 1.]])
    fig = go.Figure()

    x = x[None, ...].cuda()
    v = torch.zeros([100, 1, 2]).cuda()
    # v *= -1
    # v[..., 1] *= -1
    v[..., 1] = -1
    v_norm = v.pow(2).sum(dim=-1, keepdim=True).sqrt()
    v /= v_norm
    for i in range(100):
        v[i] *= i / 10
        # v[i, 0, 0] = (i + 1) / 10
    # v[..., 0] = torch.arange(1, 11)[..., None]

    x_next = exp(x, v)
    xs = x_next[:, 0, 0].detach().cpu().numpy()
    ys = x_next[:, 0, 1].detach().cpu().numpy()

    print(xs)
    print(ys)

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=xs,
            y=ys,
            showlegend=False
        )
    )

    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1
    )

    fig.write_image('trace.jpg')

