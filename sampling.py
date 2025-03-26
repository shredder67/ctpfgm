import torch
import numpy as np
import functools
from tqdm import tqdm
from scipy import integrate 


class Poisson(): # taken from original PFGM paper
    def __init__(self, config):
        """Construct a PFGM.

        Args:
            config: configurations
        """
        self.config = config
        self.N = config.sampling.N

    @property
    def M(self):
        return self.config.training.M

    def prior_sampling(self, shape):
        """
        Sampling initial data from p_prior on z=z_max hyperplane.
        See Section 3.3 in PFGM paper
        """

        # Sample the radius from p_radius (details in Appendix A.4 in the PFGM paper)
        max_z = self.config.sampling.z_max
        N = self.config.data.channels * self.config.data.image_size * self.config.data.image_size + 1
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=N / 2. - 0.5, b=0.5, size=shape[0])
        inverse_beta = samples_norm / (1 - samples_norm)
        # Sampling from p_radius(R) by change-of-variable
        samples_norm = np.sqrt(max_z ** 2 * inverse_beta)
        # clip the sample norm (radius)
        samples_norm = np.clip(samples_norm, 1, self.config.sampling.upper_norm)
        samples_norm = torch.from_numpy(samples_norm).cuda().view(len(samples_norm), -1)

        # Uniformly sample the angle direction
        gaussian = torch.randn(shape[0], N - 1).cuda()
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        # Radius times the angle direction
        init_samples = unit_gaussian * samples_norm

        return init_samples.float().view(len(init_samples), self.config.data.num_channels,
                                    self.config.data.image_size, self.config.data.image_size)

    def ode(self, net_fn, x, t, x_drift=None, z_drift=None, return_with_aug=False):
        z = torch.exp(t.mean())
        if self.config.sampling.vs:
            print(z)

        if x_drift is None and z_drift is None:
            x_drift, z_drift = net_fn(x, torch.ones((len(x))).cuda() * z)
            x_drift = x_drift.view(len(x_drift), -1)
        else:
            x_drift = x_drift.view(len(x_drift), -1)

        # Substitute the predicted z with the ground-truth
        # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
        z_exp = self.config.sampling.z_exp
        if z < z_exp and self.config.training.gamma > 0:
            data_dim = self.config.data.image_size * self.config.data.image_size * self.config.data.channels
            sqrt_dim = torch.sqrt(torch.tensor(data_dim))
            norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
            x_norm = self.config.training.gamma * norm_1 / (1 -norm_1)
            x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
            z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + self.config.training.gamma)

        # predicted normalized Poisson field
        v = torch.cat([x_drift, z_drift[:, None]], dim=1)

        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, :-1].view(len(x), self.config.data.num_channels,
                        self.config.data.image_size, self.config.data.image_size)
        dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
        dx_dt_prime = z * dx_dz
        if return_with_aug:
            z = z.repeat(len(x), 1) # according to def, d(x,z) = (v(x)_x * 1/v(x)_z * z, z)dt'
            return torch.cat([dx_dt_prime.reshape(len(x), -1), z], dim=1)
        else:
            return dx_dt_prime.reshape(len(x), -1)


    def forward_pz(self, config, samples_batch, m):
        """Perturbing the augmented training data. See Algorithm 2 in PFGM paper.

        Args:
        sde: An `methods.SDE` object that represents the forward SDE.
        config: configurations
        samples_batch: A mini-batch of un-augmented training data
        m: A 1D torch tensor. The exponents of (1+\tau).

        Returns:
        Perturbed samples
        """
        tau = config.training.tau
        z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
        z = z.abs()
        # Confine the norms of perturbed data.
        # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
        if config.training.restrict_M:
            idx = (z < 0.005).squeeze()
            num = int(idx.int().sum())
            restrict_m = int(self.M * 0.7)
            m[idx] = torch.rand((num,), device=samples_batch.device) * restrict_m

        data_dim = config.data.channels * config.data.image_size * config.data.image_size
        multiplier = (1+tau) ** m

        noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
        norm_m = torch.norm(noise, p=2, dim=1) * multiplier
        # Perturb z
        perturbed_z = z.squeeze() * multiplier
        # Sample uniform angle
        gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * norm_m[:, None]
        perturbation_x = perturbation_x.view_as(samples_batch)
        # Perturb x
        perturbed_x = samples_batch + perturbation_x
        # Augment the data with extra dimension z
        perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(samples_batch), -1),
                                        perturbed_z[:, None]), dim=1)
        return perturbed_samples_vec

    def perturb_x(self, config, samples_batch, z):
        """
        idea: we use predefined z from some schedule and estimate (1 + tau)^m from it
        used in x pertubation
        sample_batch: (b, c*h*w) (if with aug. z - will be dropped)
        z: (b,)

        returns:
        perturbed_samples_vec: (b, c*h*w + 1)
        """
        
        data_dim = config.data.channels * config.data.image_size * config.data.image_size
        if samples_batch.shape[1] == data_dim + 1:
            samples_batch = samples_batch[:, :-1]
        
        noise = torch.randn_like(z).abs()
        multiplier = z.squeeze() / noise

        # perturb x
        noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
        norm_m = torch.norm(noise, p=2, dim=1) * multiplier

        # Sample uniform angle
        gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * norm_m[:, None]
        perturbation_x = perturbation_x.view_as(samples_batch)
        # Perturb x
        perturbed_x = samples_batch + perturbation_x
        
        # augment data with z
        perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(samples_batch), -1),
                                        z[:, None]), dim=1)
        return perturbed_samples_vec
    
    @torch.no_grad()
    def gt_direction(self, perturbed_samples_vec, sample_batch):
        """compute the direction towards data distribution"""
        real_samples_vec = torch.cat(
          (sample_batch.reshape(len(sample_batch), -1), torch.zeros((len(sample_batch), 1)).to(sample_batch.device)), dim=1)

        data_dim =  self.config.data.image_size * self.config.data.image_size * self.config.data.channels
        gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                                dim=[-1]).sqrt()

        # For numerical stability, timing each row by its minimum value
        distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
        distance = distance ** (data_dim + 1)
        distance = distance[:, :, None]
        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
        coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
        diff = - (perturbed_samples_vec.unsqueeze(1) - real_samples_vec)

        # Calculate empirical Poisson field (N+1 dimension in the augmented space)
        gt_direction = torch.sum(coeff * diff, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)

        gt_norm = gt_direction.norm(p=2, dim=1)
        # Normalizing the N+1-dimensional Poisson field
        gt_direction /= (gt_norm.view(-1, 1) + self.config.training.gamma)
        gt_direction *= np.sqrt(data_dim)

        return gt_direction


    @torch.no_grad()
    def cm_sample(self, model, batch_size, num_steps, device):
        """consistency like algorithm to sample from distilled pfgm"""
        n_steps = self.config.sampling.CT_eval_steps
        shape = (batch_size, self.config.data.num_channels, self.config.data.image_size, self.config.data.image_size)
        sample_batch = self.prior_sampling(shape).to(device)
        t_schedule = torch.linspace(np.log(self.config.sampling.z_max), np.log(self.config.sampling.z_min), num_steps, device=device).float()
        z_schedule = t_schedule.exp()
        dt = - (t_schedule[0] - t_schedule[-1])
        sample_batch += dt * self.ode(
            model, 
            sample_batch, 
            t=t_schedule[0].repeat(batch_size, 1)
        ).reshape(shape)
        intermediate_samples = []

        for i, (z, t) in tqdm(enumerate(zip(z_schedule[1:], t_schedule[1:]))):
            # add previous sample to intermediate samples
            intermediate_samples.append(sample_batch)
            sample_batch = self.perturb_x(self.config, sample_batch, torch.tensor([z]*batch_size, device=device))[:, :-1]
            sample_batch = sample_batch.reshape(shape)
            drift = self.ode(
                model, 
                sample_batch, 
                t.repeat(batch_size, 1).to(device, dtype=torch.float32)[:, None]
            ).reshape(shape)
            dt = t - t_schedule[-1] # 1-step into x_0
            sample_batch = sample_batch + dt * drift
            assert sample_batch.shape == shape

        return sample_batch, intermediate_samples