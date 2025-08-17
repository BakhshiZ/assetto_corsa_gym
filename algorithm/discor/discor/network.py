import torch
from torch import nn
from torch.distributions import Normal


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):
    assert isinstance(input_dim, int) and isinstance(output_dim, int)
    assert isinstance(hidden_units, list) or isinstance(hidden_units, list)

    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers).apply(initialize_weights_xavier)


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class StateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_units=hidden_units)

    def forward(self, x):
        return self.net(x)


class TwinnedStateActionFunction(BaseNetwork):

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net1 = StateActionFunction(state_dim, action_dim, hidden_units)
        self.net2 = StateActionFunction(state_dim, action_dim, hidden_units)

    def forward(self, states, actions):
        assert states.dim() == 2 and actions.dim() == 2

        x = torch.cat([states, actions], dim=1)
        value1 = self.net1(x)
        value2 = self.net2(x)
        return value1, value2


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()

        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=2*action_dim,
            hidden_units=hidden_units)

    def forward(self, states):
        assert states.dim() == 2

        # Calculate means and stds of actions.
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        stds = log_stds.exp_()

        # Gaussian distributions.
        normals = Normal(means, stds)

        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)

        # Calculate entropies.
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)

class QuantileStateActionFunction(BaseNetwork): 
    """
    Critic for QR-SAC who outputs K quantiles per state-action pair (s, a)
    """
    def __init__(self, state_dim, action_dim, num_quantiles=32,
                 hidden_units=[256, 256]):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = create_linear_network(
            input_dim=state_dim+action_dim,
            # Outputs 'K' quantiles per (s, a) pair
            output_dim=num_quantiles,
            hidden_units=hidden_units
        )

    def forward(self, x):
        """
        x is concatenated [batch_size, state_dim + action_dim]
        """
        return self.net(x)

class TwinnedQuantileStateActionFunction(BaseNetwork):
    """
    Class for twin quantile networks (q1, q2) for QR-SAC
    """
    def __init__(self, state_dim, action_dim, num_quantiles=32,
                 hidden_units=[256, 256]):
        super().__init__()
        self.net1 = QuantileStateActionFunction(state_dim, action_dim, 
                                                num_quantiles, hidden_units)
        self.net2 = QuantileStateActionFunction(state_dim, action_dim, 
                                                num_quantiles, hidden_units)

    def forward(self, states, actions):
        assert states.dim() == 2 and actions.dim() == 2
        # Concatening to pass to forward of net1, net2
        x = torch.cat([states, actions], dim=1)
        # q1, q2 in form [batch_size, num_quantiles]
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2