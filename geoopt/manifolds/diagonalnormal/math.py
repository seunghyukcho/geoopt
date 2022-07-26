import torch
from math import sqrt


# x shape: (*, N, 2)
def dist(x0, x1, keepdim):
    mus0 = x0[..., 0] / sqrt(2)
    mus1 = x1[..., 0] / sqrt(2)
    stds0 = x0[..., 1]
    stds1 = x1[..., 1]

    mu_dist = (mus0 - mus1).pow(2)
    dist1 = (mu_dist + (stds0 - stds1).pow(2) + 1e-9).log() / 2
    dist2 = (mu_dist + (stds0 + stds1).pow(2) + 1e-9).log() / 2
    dist = torch.arctanh((dist1 - dist2).exp() - 1e-9) * 2

    dist = dist.pow(2)
    dist = dist.sum(dim=-1, keepdim=keepdim) + 1e-9
    dist = dist.sqrt()

    return dist  # (*, 1)


def riemannian_metric(x):
    first_entries = 1 / (x[..., 1:] + 1e-9).pow(2)
    second_entries = 2 / (x[..., 1:] + 1e-9).pow(2)

    return torch.cat((first_entries, second_entries), dim=-1)


def inv_riemannian_metric(x):
    first_entries = x[..., 1:].pow(2)
    second_entries = x[..., 1:].pow(2) / 2

    return torch.cat((first_entries, second_entries), dim=-1)


# x shape: (*, N, 2), v shape: (*, N, 2)
def exp(x, v):
    new_x1 = torch.zeros(v.size(), device=v.device)
    new_x2 = torch.zeros(v.size(), device=v.device)

    metric_tensor = riemannian_metric(x)
    v_norm = v.pow(2) * metric_tensor
    v_norm = v_norm.sum(dim=-1, keepdim=True)

    x0 = x[..., :1]
    x1 = x[..., 1:]
    v0 = v[..., :1]
    v1 = v[..., 1:]

    # sign = torch.where((v0 > 0) + (v1 > 0), 1, -1)
    sign1 = torch.where(v0 > 0, 1, -1)
    sign2 = torch.where(v1 > 0, 1, -1)
    r1 = sign1 * (v_norm / 2).sqrt()
    r2 = sign2 * (v_norm / 2).sqrt()

    new_x1[..., :1] = x0
    new_x1[..., 1:] = x1 * r2.exp()

    p = x0 + 2 * v1 * x1 / (v0 + 1e-9)
    b = ((x0 - p).pow(2) / 2 + x1.pow(2)).sqrt()

    t0 = torch.arctanh((x0 - p) / (sqrt(2) * b)) / r1
    new_x2[..., :1] = p + sqrt(2) * b * torch.tanh(r1 * (t0 + 1))
    new_x2[..., 1:] = b / torch.cosh(r1 * (t0 + 1))

    new_x = torch.where(v0 == 0, new_x1, new_x2)

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

