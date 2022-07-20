import torch
from math import sqrt


# x shape: (*, N, 2)
def dist(x0, x1, keepdim):
    mus0 = x0[..., 0] / sqrt(2)
    mus1 = x1[..., 0] / sqrt(2)
    stds0 = x0[..., 1]
    stds1 = x1[..., 1]

    mu_dist = (mus0 - mus1).pow(2)
    dist1 = (mu_dist + (stds0 + stds1).pow(2) + 1e-9).sqrt()
    dist2 = (mu_dist + (stds0 - stds1).pow(2) + 1e-9).sqrt()
    dist_l = (dist1 + dist2 + 1e-9).log()
    dist_r = (dist1 - dist2 + 1e-9).log()
    dist = dist_l - dist_r
    dist = dist.pow(2)
    dist = dist.sum(dim=-1, keepdim=keepdim)
    dist = dist * 2 + 1e-9
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

    r = (v_norm / 2).sqrt()

    x0 = x[..., :1]
    x1 = x[..., 1:]
    v0 = v[..., :1]
    v1 = v[..., 1:]

    sign = torch.where(v1 > 0, 1, -1)
    new_x1[..., :1] = x0
    new_x1[..., 1:] = x1 * (r * sign).exp()

    p = x0 + 2 * v1 * x1 / v0
    b = ((x0 - p).pow(2) / 2 + x1.pow(2)).sqrt()

    t0 = torch.arctanh((x0 - p) / (sqrt(2) * b)) / r
    sign = torch.where(v0 > 0, 1, -1)
    new_x2[..., :1] = p + sqrt(2) * b * torch.tanh(r * (t0 + sign))
    new_x2[..., 1:] = b / torch.cosh(r * (t0 + sign))

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

