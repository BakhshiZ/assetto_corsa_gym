import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.optim import Adam

from discor.network import (
    GaussianPolicy,
    TwinnedQuantileStateActionFunction
)


# Helper functions
def soft_update(source: torch.nn.Module,
                target: torch.nn.Module,
                polyak_avg_coeff: float):
    with torch.no_grad():
        # Function to perform Polyak (soft) update of target network params
        # Moves a fraction of way towards source with every update
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            """
            Polyak formula -> theta_tar = (1 - tau) * theta_tar + tau * theta_source
            """
            target_param.mul_(1.0 - polyak_avg_coeff).add_(source_param, alpha=polyak_avg_coeff)

def copy_params(source: torch.nn.Module,
                target: torch.nn.Module):
    with torch.no_grad():
        # Function to make target params an exact copy of source immediately
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.copy_(source_param)

def quantile_huber_loss(
        predicted_q: torch.Tensor,
        target_q: torch.Tensor,
        kappa: float):
    
    # Helper function for running quantile huber loss
    batch_size, num_quantiles = predicted_q.shape

    # Making tensor in form of [batch_size, num_quantiles, num_quantiles]
    temporal_diff_err = target_q.unsqueeze(1) - predicted_q.unsqueeze(2)
    abs_temp_diff_err = temporal_diff_err.abs()

    # Applying huber loss formula (MSE for small error, MAE for large)
    huber_loss = torch.where(
        abs_temp_diff_err <= kappa,
        # MSE
        0.5 * (temporal_diff_err ** 2),
        # MAE
        kappa * (abs_temp_diff_err - 0.5 * kappa)
    )

    # Computing quantile midpoints
    quantile_mps = (torch.arange(
                            num_quantiles, 
                            device=predicted_q.device, 
                            dtype=predicted_q.dtype) + 0.5)
    quantile_mps /= num_quantiles
    # Reshaping to prevent errors
    quantile_mps = quantile_mps.view(1, num_quantiles, 1)

    # Matrix of 1s and 0s to check for overshooting
    indicator = (temporal_diff_err < 0.0).float()

    # Giving less weight to quantiles that overshot
    quantile_weight = (quantile_mps - indicator).abs()

    loss = (quantile_weight * huber_loss).sum(dim=2).mean(dim=1).mean()
    return loss

# Config dataclass
@dataclass
class QRSAC_Config:
    state_dim: int
    action_dim: int
    device: str
    hidden_units: Optional[List[int]] = None
    num_quantiles: int = 32
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    autotune_alpha: bool = True
    target_entropy: Optional[float] = None
    quantile_huber_kappa: float = 1.0
    nstep: int = 1

    # In case there are issues after running __init__
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [256, 256]
        
        assert self.num_quantiles > 0
        assert 0.0 < self.tau <= 1.0
        assert 0.0 < self.gamma < 1.0


class QRSAC:
    def __init__(self, cfg: QRSAC_Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = cfg.gamma
        self.nstep = cfg.nstep
        self.update_entropy = True
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim

        # Setting up networks
        self.policy_net = GaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_units=cfg.hidden_units
        ).to(self.device)

        self.critics_net = TwinnedQuantileStateActionFunction(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_quantiles=cfg.num_quantiles,
            hidden_units=cfg.hidden_units
        ).to(self.device)

        self.target_critics_net = TwinnedQuantileStateActionFunction(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_quantiles=cfg.num_quantiles,
            hidden_units=cfg.hidden_units
        ).to(self.device)

        # Initially both should be same
        copy_params(self.critics_net, self.target_critics_net)

        # Setting up optimizers
        self.actor_opt = Adam(self.policy_net.parameters(), lr=cfg.actor_lr)
        self.critic_opt = Adam(self.critics_net.parameters(), lr=cfg.critic_lr)

        # If alpha is modifiable
        if cfg.autotune_alpha:
            # From SAC paper, use -action_dim if tar_entropy not specified
            target_entropy = -float(self.action_dim) if cfg.target_entropy is None else float(cfg.target_entropy)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = target_entropy
            self.autotune_alpha = True
        else:
            # Fix alpha at 1.0
            self.log_alpha = torch.tensor(0.0, device=self.device)
            self.alpha_opt = None
            self.target_entropy = None
            self.autotune_alpha = False

    def _alpha(self):
        # logx^x = x
        return self.log_alpha.exp()
    
    @torch.no_grad()
    def explore(self, state):
        s_tensor = torch.as_tensor(data=state, 
                                   device=self.device,
                                   dtype=torch.float32).unsqueeze(0)
        action, entropies, _ = self.policy_net(s_tensor)
        # Converting action tensor to cpu so numpy can read it
        return action.cpu().numpy()[0], entropies
    
    @torch.no_grad()
    def exploit(self, state):
        s_tensor = torch.as_tensor(data=state,
                                   device=self.device,
                                   dtype=torch.float32).unsqueeze(0)
        _, _, mean_action = self.policy_net(s_tensor)
        # Return mean action (no noise) for exploiting
        return mean_action.cpu().numpy()[0], torch.zeros(1, 1, device=self.device)

    def update_online_networks(self, buffer_data, writer=None):
        """
        Main training step:
            * update actor
            * update critics
            * update entropy temperature (maybe)
        """
        states, actions, rewards, next_states, dones = buffer_data
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)  # Shape [batch_size, 1]
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)  # Shape [batch_size, 1]

        alpha = self._alpha()

        # Computing Bellman Target Distribution that critics should match
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy_net(next_states)          # next_entropies: [B,1]
            target_q1, target_q2 = self.target_critics_net(next_states, next_actions)  # [B,K], [B,K]
            target_min = torch.min(target_q1, target_q2) + alpha * next_entropies      # [B,K] + [B,1] -> [B,K]
            target_return_distbn = rewards + (1 - dones) * self.gamma * target_min     # [B,1] + [B,K] -> [B,K]

        # Calculating loss for critics
        q1, q2 = self.critics_net(states, actions)

        # After q1, q2 and target_return_distbn are computed
        assert q1.dim() == 2 and q2.dim() == 2, f"Critic outputs must be [B,K], got {q1.shape}, {q2.shape}"
        assert target_return_distbn.dim() == 2, f"Target must be [B,K], got {target_return_distbn.shape}"
        assert q1.shape[1] == target_return_distbn.shape[1], \
            f"num_quantiles mismatch: predicted {q1.shape[1]} vs target {target_return_distbn.shape[1]}"


        critic1_loss = quantile_huber_loss(q1, target_return_distbn, self.cfg.quantile_huber_kappa)
        critic2_loss = quantile_huber_loss(q2, target_return_distbn, self.cfg.quantile_huber_kappa)
        critic_total_loss = critic1_loss + critic2_loss

        """
        standard three-step PyTorch optimization loop for updating critics
            * Zero gradients from previous backward passes
            * Backpropagate loss
            * Take optimizer step
        """

        self.critic_opt.zero_grad(set_to_none=True)
        critic_total_loss.backward()
        self.critic_opt.step()

        # Updating actor
        actor_actions, entropies, _ = self.policy_net(states)
        q1_actor, q2_actor = self.critics_net(states, actor_actions)
        q_actor = torch.min(q1_actor, q2_actor).mean(dim=1, keepdim=True)
        actor_loss = (-alpha * entropies - q_actor).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Updating alpha
        alpha_loss_val = 0.0
        alpha_val = alpha.detach().item()
        
        if self.autotune_alpha and self.update_entropy:
            alpha_loss = (self._alpha() * (-entropies - self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = alpha_loss.detach().item()
            alpha_val = self._alpha().detach().item()
        
        return {
            "critic1_quantile_loss": critic1_loss.item(),
            "critic2_quantile_loss": critic2_loss.item(),
            "critic_loss": critic_total_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha_val,
            "alpha_loss": alpha_loss_val,
            "entropy_mean": entropies.mean().item(),
            "q_pi_mean": q_actor.mean().item(),
        }
    
    def update_target_networks(self):
        soft_update(self.critics_net, self.target_critics_net, self.cfg.tau)

    def save_models(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)  # ensure folder exists
        torch.save(self.policy_net.state_dict(), os.path.join(dir_path, "policy.pth"))
        torch.save(self.critics_net.state_dict(), os.path.join(dir_path, "critics.pth"))
        torch.save(self.target_critics_net.state_dict(), os.path.join(dir_path, "target_critics.pth"))
        
        alpha_ckpt = {
            "log_alpha": self.log_alpha.detach(),
            "autotune": self.autotune_alpha,
            "target_entropy": self.target_entropy
        }
        torch.save(alpha_ckpt, os.path.join(dir_path, "alpha.pth"))


    def load_models(self, dir_path):
        # Try to load QR-SAC format first, fallback to SAC format
        policy_path = os.path.join(dir_path, "policy.pth")
        if not os.path.exists(policy_path):
            # Fallback to SAC format
            policy_path = os.path.join(dir_path, "policy_net.pth")
            if os.path.exists(policy_path):
                print(f"Loading SAC policy checkpoint: {policy_path}")
            else:
                raise FileNotFoundError(f"Neither policy.pth nor policy_net.pth found in {dir_path}")
        
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
        
        # Try to load QR-SAC critics, skip if not available (SAC checkpoint)
        critics_path = os.path.join(dir_path, "critics.pth")
        if os.path.exists(critics_path):
            self.critics_net.load_state_dict(torch.load(critics_path, map_location=self.device))
            target_critics_path = os.path.join(dir_path, "target_critics.pth")
            if os.path.exists(target_critics_path):
                self.target_critics_net.load_state_dict(torch.load(target_critics_path, map_location=self.device))
        else:
            print("QR-SAC critics not found, initializing randomly (loading from SAC checkpoint)")
        
        # Load alpha parameters if available
        alpha_path = os.path.join(dir_path, "alpha.pth")
        if os.path.exists(alpha_path):
            alpha_ckpt = torch.load(alpha_path, map_location=self.device)
            self.log_alpha = alpha_ckpt.get("log_alpha", torch.tensor(0.0)).to(self.device)
            self.log_alpha.requires_grad_(True)
            self.autotune_alpha = bool(alpha_ckpt.get("autotune", True))
            self.target_entropy = alpha_ckpt.get("target_entropy", -float(self.action_dim))
            if self.autotune_alpha and self.alpha_opt is None:
                self.alpha_opt = Adam([self.log_alpha], lr=self.cfg.alpha_lr)
