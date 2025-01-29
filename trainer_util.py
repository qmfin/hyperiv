import numpy as np
import torch
from tqdm import tqdm


def get_grad(
    model,
    z_batch,
    k_start=-1.5,
    k_end=0.5,
    k_num=100,
    t_start=0.01,
    t_end=2.0,
    t_num=25,
):
    batch_size = z_batch.shape[0]
    k_samples = torch.linspace(
        k_start, k_end, k_num, device=z_batch.device, requires_grad=True
    )
    t_samples = torch.linspace(
        t_start, t_end, t_num, device=z_batch.device, requires_grad=True
    )
    kt = torch.cartesian_prod(k_samples, t_samples)
    kt = torch.tile(kt, (batch_size, 1, 1))
    pred = model(z_batch, kt)
    grad = torch.autograd.grad(pred, kt, torch.ones_like(pred), create_graph=True)[0]
    grad_k, grad_t = grad[..., 0], grad[..., 1]
    grad2 = torch.autograd.grad(
        grad_k,
        kt,
        grad_outputs=torch.ones_like(grad_k),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad2_kk = grad2[..., 0]
    return kt, pred[..., 0], grad_k, grad_t, grad2_kk


def calc_cal_loss(t, s, s_t):
    cond = s + 2 * t * s_t
    loss = torch.relu(-cond).mean()
    return loss


def calc_g_func(k, t, s, s_k, s_kk):
    term1 = (1 - k * s_k / s) ** 2
    term2 = -((0.5 * t * s * s_k) ** 2)
    term3 = t * s * s_kk
    return term1 + term2 + term3


def calc_density(k, t, s, g):
    d_minus = (-k - 0.5 * s**2 * t) / (s * t**0.5)
    p = g * torch.exp(-0.5 * d_minus**2) / (s * torch.sqrt(2 * torch.pi * t))
    return p


def aux_loss(
    model, z_batch, k_start=-1.5, k_end=0.5, k_num=71, t_start=0.01, t_end=2.0, t_num=17
):
    kt, s, s_k, s_t, s_kk = get_grad(
        model, z_batch, k_start, k_end, k_num, t_start, t_end, t_num
    )
    k, t = kt[..., 0], kt[..., 1]
    cal_loss = calc_cal_loss(t, s, s_t)
    g = calc_g_func(k, t, s, s_k, s_kk)
    g_loss = torch.relu(-g).mean()
    p = calc_density(k, t, s, g)
    integral = torch.trapz(p.view(-1, k_num, t_num), k.view(-1, k_num, t_num), dim=1)
    integral_loss = ((integral - 1.0) ** 2).mean()
    return cal_loss, g_loss, integral_loss


def trainer(dataloader, model, device, optimizer, is_train=True):
    criterion = torch.nn.MSELoss()
    model.train() if is_train else model.eval()
    total_loss_mse, total_loss_cal, total_loss_g, total_loss_integral = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    all_y_true, all_y_pred = [], []

    for batch in tqdm(dataloader):
        z_batch, X_batch, y_batch = [x.to(device) for x in batch]
        if is_train:
            optimizer.zero_grad()

        y_pred = model(z_batch, X_batch).squeeze(-1)
        all_y_true.append(y_batch.reshape(-1).detach().cpu().numpy())
        all_y_pred.append(y_pred.reshape(-1).detach().cpu().numpy())

        mse_loss = criterion(y_pred, y_batch)
        cal_loss, g_loss, integral_loss = aux_loss(model, z_batch)
        loss = mse_loss + cal_loss + g_loss + integral_loss

        if is_train:
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
            optimizer.step()

        total_loss_mse += mse_loss.item()
        total_loss_cal += cal_loss.item()
        total_loss_g += g_loss.item()
        total_loss_integral += integral_loss.item()

    total_loss_mse /= len(dataloader)
    total_loss_cal /= len(dataloader)
    total_loss_g /= len(dataloader)
    total_loss_integral /= len(dataloader)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    total_loss_mae = np.mean(np.abs(all_y_true - all_y_pred))

    return (
        total_loss_mse,
        total_loss_mae,
        total_loss_cal,
        total_loss_g,
        total_loss_integral,
    )
