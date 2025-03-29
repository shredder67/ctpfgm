import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import click
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from tqdm import tqdm
import wandb
from diffusers.utils import make_image_grid
from PIL import Image
from ml_collections import ConfigDict

from data import CMNISTDataset
from model import DDPM, ExponentialMovingAverage
from utils import seed_everything, get_mnist_config, pseudo_huber_loss, optimization_manager
from sampling import Poisson

import json
import os

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('config.json', 'r') as f:
    private_config = json.load(f)
os.environ["WANDB_API_KEY"] = private_config['wandb_key']


def seed_everything(seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def simple_precond(sde, x, model_pred, z_min, z_max):
    """dummy linear parametrization to meet the consistency property"""
    x_x, x_z = x[:, :-1], x[:, -1]
    x_x_dir, x_z_dir = model_pred[:, :-1], model_pred[:, -1]
    coef = torch.clamp((x_z - z_min)/ (z_max - z_min), 0, 1)[:, None] # 0 - close to data, 1 - close to prior
    u = (1 - coef) * x + coef * model_pred
    return u


def loss_ect(sde, model, ema, samples_batch, dt = 0.01):
    raise NotImplementedError


def loss_ct(sde, model, ema, samples_batch, dt = 0.01):
    samples_full = samples_batch 
    samples_batch = samples_batch[: sde.config.training.small_batch_size]


    # 1. perturb samples to some z (following empirical pertub alg. from pfgm)
    m = torch.rand((samples_batch.shape[0],), device=samples_batch.device) * sde.M
    perturbed_samples_vec = sde.forward_pz(sde.config, samples_batch, m)


    with torch.no_grad():
        # 2. make a single step towards data distribution via ODE with empirical E(x)
        empirical_dir = sde.gt_direction(perturbed_samples_vec, samples_full)

        # euler ode step towards data points
        drift = sde.ode(
            model, 
            perturbed_samples_vec[:, :-1].view_as(samples_batch), 
            t=torch.log(perturbed_samples_vec[:, -1]),
            x_drift=empirical_dir[:, :-1],
            z_drift=empirical_dir[:, -1].squeeze(),
            return_with_aug=True
        )
        if sde.config.training.ct_dt_mode == 'fixed':
            num_steps = sde.config.sampling.CT_steps
            dt = - (np.log(sde.config.sampling.z_max) - np.log(sde.config.sampling.z_min)) / num_steps
        else:
            raise ValueError(f"CT_dt_mode {sde.config.training.ct_dt_mode} not supported")

        attracted_samples_vec = perturbed_samples_vec + dt * drift

        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        target_dir_x, target_dir_z = model(
            attracted_samples_vec[:, :-1].view_as(samples_batch),
            attracted_samples_vec[:, -1]
        )
        target_dir_x = target_dir_x.view(len(target_dir_x), -1)
        target_dir = torch.cat([target_dir_x, target_dir_z[:, None]], dim=1).detach()
        ema.restore(model.parameters())
    
    # 3. loss between this directions from two different points
    pred_dir_x, pred_dir_z = model(
        perturbed_samples_vec[:, :-1].view_as(samples_batch),
        perturbed_samples_vec[:, -1]
    )
    pred_dir_x = pred_dir_x.view(len(pred_dir_x), -1)
    pred_dir = torch.cat([pred_dir_x, pred_dir_z[:, None]], dim=1)

    loss_fn = mse_loss if sde.config.training.ct_loss_type == 'mse' else pseudo_huber_loss
    if sde.config.training.ct_precond == 'simple':
        with torch.no_grad():
            target_u = simple_precond(sde, attracted_samples_vec, target_dir, sde.config.sampling.z_min, sde.config.sampling.z_max)
        pred_u = simple_precond(sde, perturbed_samples_vec, pred_dir, sde.config.sampling.z_min, sde.config.sampling.z_max)
        loss = loss_fn(pred_u, target_u)
    else:
        loss = loss_fn(pred_dir, target_dir)
    
    if sde.config.training.ct_loss_type == 'pseudo_huber':
        loss = loss.mean()

    return loss


@click.command()
@click.option("--alg", default="ct", help="consistency training algorithm (ct, ect)")
def main(alg):
    seed_everything(SEED)
    config = get_mnist_config()
    batch_size = config.training.batch_size

    # 0. prepare data
    train_dataset = CMNISTDataset(train=True)
    test_dataset = CMNISTDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 1. prepare model
    model = DDPM(config).to(DEVICE)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay
    )
    optim_fn = optimization_manager(config)
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    sde = Poisson(config)

    wandb.finish()
    run = wandb.init(
        project="ct-cmnist", 
        config=config,
    )

    if alg == "ct":
        step = 0
        epoch = 0
        while step < config.training.n_iters:
            for batch in tqdm(train_loader, desc="training, epoch %d, step %d" % (epoch, step)):
                batch = batch.to(DEVICE)
                loss = loss_ct(sde, model, ema, batch)
                loss.backward()
                optim_fn(optimizer, model.parameters(), step)
                step += 1
                state['ema'].update(model.parameters())
                wandb.log({"train/loss":loss.item()},step=step)

                if step % config.training.eval_freq == 0:
                    try:
                        test_batch = next(test_loader_iter)
                    except (NameError, StopIteration):
                        test_loader_iter = iter(test_loader)
                        test_batch = next(test_loader_iter)
                    test_batch = test_batch.to(DEVICE)
                    with torch.no_grad():
                        eval_loss = loss_ct(sde, model, ema, test_batch)
                        wandb.log({"eval/loss":eval_loss.item()},step=step)

                    # sample images and log to wandb
                    sample_batch, intermediate_samples = sde.cm_sample(model, batch_size=8, num_steps=config.sampling.CT_eval_steps, device=DEVICE)
                    sample_batch = np.clip(sample_batch.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    sample_images = [Image.fromarray(sample) for sample in sample_batch]
                    
                    wandb.log({"eval/samples": [wandb.Image(make_image_grid(sample_images, 2, 4))]}, step=step)
                    for i, sample in enumerate(intermediate_samples):
                        if i in [0, len(intermediate_samples)//2, len(intermediate_samples)-1]:
                            sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                            sample_images = [Image.fromarray(sample) for sample in sample]
                            wandb.log({f"eval/intermediate_samples/step_{i+1}": [wandb.Image(make_image_grid(sample_images, 2, 4))]}, step=step)

                if step % config.training.snapshot_freq == 0:
                    wandb_run_name = run.name
                    if not os.path.exists("checkpoints"):
                        os.makedirs("checkpoints")
                    ckpt_dir = f"checkpoints/ct-cmnist-{config.model.name}-{wandb_run_name}-{step}.pt"
                    saved_state = {
                        'optimizer': state['optimizer'].state_dict(),
                        'model': state['model'].state_dict(),
                        'ema': state['ema'].state_dict(),
                        'step': state['step']
                    }
                    torch.save(saved_state, ckpt_dir)


    elif alg == "ect":
        raise NotImplementedError
    else:
        raise ValueError(f"Algorithm {alg} not supported")


if __name__ == "__main__":
    main()