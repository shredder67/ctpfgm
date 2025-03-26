import torch
import numpy as np 
import math
from ml_collections import ConfigDict


def seed_everything(seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def pseudo_huber_loss(input, target):
    """used in iCT"""
    c = 0.00054 * math.sqrt(math.prod(input.shape[1:]))
    return torch.sqrt((input - target) ** 2 + c**2) - c


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_mnist_config():
    config = ConfigDict()

    config.device = 'cuda'

    config.data = ConfigDict()
    config.data.num_channels = 3
    config.data.channels = 3
    config.data.centered = True
    config.data.img_resize=16
    config.data.image_size =16


    config.training = ConfigDict()
    config.training.sde = 'poisson'
    config.training.continuous = True
    config.training.batch_size = 4096 #1024
    config.training.small_batch_size = 512
    config.training.gamma = 5
    config.training.restrict_M = True
    config.training.tau = 0.03
    config.training.snapshot_freq = 5_000
    config.training.eval_freq = 1_000
    config.training.model = 'ddpmpp'
    config.training.M = 291
    config.training.reduce_mean = False
    config.training.n_iters =  1_000_000
    config.training.fid_freq = 25_000
    config.training.fid_batch_size = 250
    config.training.ct_dt_mode = 'fixed' # 'fixed' or 'adaptive'
    config.training.ct_loss_type = 'mse' # 'mse', 'pseudo_huber'

    config.model  = ConfigDict()
    config.model.name = 'ncsnpp'
    config.model.scale_by_sigma = False
    config.model.ema_rate = 0.9999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = 128
    config.model.ch_mult = (1, 2, 2, 2)
    config.model.num_res_blocks = 4
    config.model.attn_resolutions = (16,)
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.fir = False
    config.model.fir_kernel = [1, 3, 3, 1]
    config.model.skip_rescale = True
    config.model.resblock_type = 'biggan'
    config.model.progressive = 'none'
    config.model.progressive_input = 'none'
    config.model.progressive_combine = 'sum'
    config.model.attention_type = 'ddpm'
    config.model.init_scale = 0.
    config.model.fourier_scale = 16
    config.model.embedding_type = 'positional'
    config.model.conv_size = 3
    config.model.sigma_end = 0.01
    config.model.dropout = 0.3 # see https://arxiv.org/pdf/2310.14189 on increased dropout

    config.optim  = ConfigDict()
    config.optim.weight_decay = 0.9999
    config.optim.optimizer = 'Adam'
    config.optim.lr = 1e-5
    config.optim.beta1 = 0.9
    config.optim.beta2 = 0.999
    config.optim.eps = 1e-8
    config.optim.warmup = 5000
    config.optim.grad_clip = 1.


    config.device = 'cuda'

    config.sampling = ConfigDict()
    config.sampling.method = 'ode'
    config.sampling.ode_solver = 'rk45'
    config.sampling.N = 100
    config.sampling.CT_steps = 1000 # disrc. numbe rfor consistency training
    config.sampling.CT_relative_discr = 100 # relative discretization for CT (1/N for left distance towards data)
    config.sampling.CT_eval_steps = 8 # used during validation sampling
    config.sampling.z_max = 30
    config.sampling.z_min = 1e-7
    config.sampling.upper_norm = 3000
    config.sampling.z_exp=1
    config.sampling.visual_iterations = 10
    # verbose
    config.sampling.vs = False

    return config