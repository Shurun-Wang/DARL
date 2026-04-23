# -*- coding: utf-8 -*-
from utils import safe_forward_check
from models.get_model import get_model, get_opt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from utils import get_model_complexity_info
from models.complexity import COMPLEXITY_BASELINES


def train_one_epoch_adversarial(model, device, data_loader, criterion, optimizer, adv_cfg):
    model.train()
    eff_grad_noise = adv_cfg['grad_noise']

    for x, y in data_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # --- Attack 3: Gradient Noise ---
        if eff_grad_noise > 0:
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * eff_grad_noise
                    param.grad.add_(noise)
        optimizer.step()


def evaluate(model, device, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for samples, targets in data_loader:
            samples, targets = samples.to(device), targets.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return (correct / total) if total > 0 else 0.0


def train_child_phased_safe(args, builder_cfg, destroyer_cfg, ch_num, train_loader, val_loader,
                            device, epochs, invalid_reward=-1.0):
    try:
        model = get_model(args.model_name, ch_num, builder_cfg).to(device)
        optimizer = get_opt(model.parameters(), builder_cfg)
    except Exception as e:
        return invalid_reward, 0.0, 0.0 # Acc, Params, Flops

    if not safe_forward_check(model, device, train_loader):
        return invalid_reward, 0.0, 0.0 # Acc, Params, Flops

    params_m, flops_g = get_model_complexity_info(model, input_shape=(1, ch_num, 2000), device=device)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    for _ in range(epochs):
        train_one_epoch_adversarial(model, device, train_loader, criterion, optimizer, destroyer_cfg)
        acc = evaluate(model, device, val_loader)
        if acc > best_val_acc:
            best_val_acc = acc
    return float(best_val_acc), params_m, flops_g


def compute_gae(rewards, values, device, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T).to(device)
    gae = 0
    next_value = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


class PPOController(nn.Module):
    def __init__(self, search_space, hidden_size=128, embed_size=32):
        super().__init__()
        self.search_space = search_space
        self.keys = list(search_space.keys())
        self.action_dims = [len(search_space[k]) for k in self.keys]
        self.actor_lstm = nn.LSTMCell(embed_size, hidden_size)
        self.actor_heads = nn.ModuleDict(
            {k: nn.Linear(hidden_size, dim) for k, dim in zip(self.keys, self.action_dims)})
        self.critic_lstm = nn.LSTMCell(embed_size, hidden_size)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.action_embed = nn.Embedding(sum(self.action_dims), embed_size)
        self.action_offset = {}
        idx = 0
        for k, dim in zip(self.keys, self.action_dims):
            self.action_offset[k] = idx
            idx += dim
        self.init_input = nn.Parameter(torch.randn(1, embed_size))

    def forward(self, device="cpu", old_actions=None):
        batch_size = 1
        h_a, c_a = torch.zeros(batch_size, 128, device=device), torch.zeros(batch_size, 128, device=device)
        h_c, c_c = torch.zeros(batch_size, 128, device=device), torch.zeros(batch_size, 128, device=device)
        x = self.init_input.repeat(batch_size, 1)

        log_probs, actions, entropies, values = [], [], [], []

        for step, k in enumerate(self.keys):
            h_a, c_a = self.actor_lstm(x, (h_a, c_a))
            logits = self.actor_heads[k](h_a)
            probs = F.softmax(logits, dim=-1)
            log_probs_k = F.log_softmax(logits, dim=-1)

            if old_actions is not None:
                action_idx = torch.tensor([old_actions[step]], device=device)
            else:
                action_idx = torch.distributions.Categorical(probs).sample()

            log_probs.append(log_probs_k.gather(1, action_idx.unsqueeze(-1)))
            entropies.append(-(probs * log_probs_k).sum(-1, keepdim=True))
            actions.append(action_idx.item())

            h_c, c_c = self.critic_lstm(x, (h_c, c_c))
            values.append(self.critic_head(h_c))

            global_idx = self.action_offset[k] + action_idx.item()
            x = self.action_embed(torch.tensor([global_idx], device=device))

        cfg = {k: self.search_space[k][actions[i]] for i, k in enumerate(self.keys)}
        return {'actions': actions, 'log_probs': torch.cat(log_probs), 'entropies': torch.cat(entropies),
                'values': torch.cat(values).squeeze(-1), 'config': cfg}

    def evaluate_actions(self, stored_actions, device="cpu"):
        return self.forward(device=device, old_actions=stored_actions)


class ContinuousPPOController(nn.Module):
    def __init__(self, search_space, hidden_size=128, embed_size=32):
        super().__init__()
        self.search_space = search_space
        self.keys = list(search_space.keys())

        self.actor_lstm = nn.LSTMCell(embed_size, hidden_size)
        self.mu_heads = nn.ModuleDict({k: nn.Linear(hidden_size, 1) for k in self.keys})
        self.log_stds = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in self.keys})

        self.critic_lstm = nn.LSTMCell(embed_size, hidden_size)
        self.critic_head = nn.Linear(hidden_size, 1)

        self.action_embed = nn.Linear(1, embed_size)
        self.init_input = nn.Parameter(torch.randn(1, embed_size))

    def forward(self, device="cpu", old_actions=None):
        batch_size = 1
        h_a, c_a = torch.zeros(batch_size, 128, device=device), torch.zeros(batch_size, 128, device=device)
        h_c, c_c = torch.zeros(batch_size, 128, device=device), torch.zeros(batch_size, 128, device=device)
        x = self.init_input.repeat(batch_size, 1)

        log_probs, actions, entropies, values = [], [], [], []
        cfg = {}

        for step, k in enumerate(self.keys):
            h_a, c_a = self.actor_lstm(x, (h_a, c_a))
            mu = self.mu_heads[k](h_a)  # [1, 1]
            std = torch.exp(self.log_stds[k]).expand_as(mu)  # [1, 1]

            dist = torch.distributions.Normal(mu, std)

            if old_actions is not None:
                action_raw = torch.tensor([[old_actions[step]]], device=device, dtype=torch.float32)
            else:
                action_raw = dist.sample()

            log_prob = dist.log_prob(action_raw)
            entropy = dist.entropy()

            log_probs.append(log_prob.view(-1))
            entropies.append(entropy.view(-1))
            actions.append(action_raw.item())

            h_c, c_c = self.critic_lstm(x, (h_c, c_c))
            values.append(self.critic_head(h_c).view(-1))

            x = self.action_embed(action_raw)

            action_bounded = torch.sigmoid(action_raw).item()
            min_val, max_val = self.search_space[k]
            cfg[k] = min_val + (max_val - min_val) * action_bounded

        return {
            'actions': actions,
            'log_probs': torch.cat(log_probs),
            'entropies': torch.cat(entropies),
            'values': torch.cat(values),
            'config': cfg
        }

    def evaluate_actions(self, stored_actions, device="cpu"):
        return self.forward(device=device, old_actions=stored_actions)


class PPO:
    def __init__(self, controller, device, lr=3e-4, batch_size=8):
        self.controller = controller
        self.optimizer = optim.AdamW(controller.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
        self.memory = []

    def store_experience(self, actions, log_probs, values, reward):
        self.memory.append(
            {'actions': actions, 'log_probs': log_probs.detach(), 'values': values.detach(), 'reward': reward})

    def update(self):
        if len(self.memory) < self.batch_size: return 0.0
        old_log_probs = torch.cat([m['log_probs'] for m in self.memory])
        rewards = torch.tensor([m['reward'] for m in self.memory])
        values = torch.stack([m['values'] for m in self.memory])
        rewards = rewards.repeat_interleave(len(self.controller.keys)).reshape(values.shape)

        batch_adv, batch_ret = torch.zeros_like(rewards).to(self.device), torch.zeros_like(rewards)
        for i in range(rewards.shape[0]):
            batch_adv[i], batch_ret[i] = compute_gae(rewards[i], values[i], self.device, gamma=1, lam=1)

        batch_adv = batch_adv.view(-1, 1)
        batch_ret = batch_ret.view(-1).to(self.device)

        total_loss = 0
        for _ in range(3):
            new_log_probs, new_ents, new_vals = [], [], []
            for m in self.memory:
                res = self.controller.evaluate_actions(m['actions'], self.device)
                new_log_probs.append(res['log_probs'])
                new_ents.append(res['entropies'])
                new_vals.append(res['values'])

            new_log_probs = torch.cat(new_log_probs)
            new_vals = torch.cat(new_vals)
            new_ents = torch.cat(new_ents)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_adv

            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(new_vals, batch_ret) - 0.01 * new_ents.mean()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
            self.optimizer.step()

        self.memory.clear()
        return total_loss / 3


def darl_optimizer(args, ch_num, device, train_loader, val_loader, search_dict):
    W_PARAMS = 0.05
    W_FLOPS = 0.05
    comp = COMPLEXITY_BASELINES[args.dataset][args.model_name]
    param_base, flop_base = comp['param'], comp['flop']
    time1 = time.time()

    destroyer_space = {'grad_noise': (0.0, 0.01)}

    builder_ctrl = PPOController(search_dict).to(device)
    builder_ppo = PPO(builder_ctrl, device, batch_size=4)

    destroyer_ctrl = ContinuousPPOController(destroyer_space).to(device)
    destroyer_ppo = PPO(destroyer_ctrl, device, batch_size=4)
    total_episodes = args.search_trials
    history = []

    destroyer_strategy_history = {}
    stats = {"total_trials": 0, "success_trials": 0, "failed_trials": 0}
    successful_hp_set = set()

    for episode in range(1, total_episodes + 1):
        stats["total_trials"] += 1
        with torch.no_grad():
            b_res = builder_ctrl.forward(device=device)
            d_res = destroyer_ctrl.forward(device=device)

        acc, param, flop = train_child_phased_safe(
            args, b_res['config'], d_res['config'], ch_num, train_loader, val_loader, device, epochs=args.search_epochs)

        if acc == -1.0:
            stats["failed_trials"] += 1
            r_builder = -1.0
        else:
            stats["success_trials"] += 1
            config_tuple = tuple(sorted(b_res['config'].items()))
            successful_hp_set.add(config_tuple)
            destroyer_strategy_history[episode] = {'grad_noise': d_res['config']['grad_noise']}

            pen_p = float(np.log10(1 + param / param_base))
            pen_f = float(np.log10(1 + flop / flop_base))
            total_penalty = W_PARAMS * pen_p + W_FLOPS * pen_f
            efficiency_factor = 1.0 - total_penalty
            r_builder = acc * efficiency_factor

        builder_ppo.store_experience(b_res['actions'], b_res['log_probs'], b_res['values'], r_builder)
        b_loss = builder_ppo.update()

        r_destroyer = -1
        d_loss = -1
        if acc != -1.0:
            r_destroyer = (1 - acc) * float(np.exp(d_res['config']['grad_noise']))
            destroyer_ppo.store_experience(d_res['actions'], d_res['log_probs'], d_res['values'], r_destroyer)
            d_loss = destroyer_ppo.update()

        history.append({
            "trial": episode,
            "reward": float(r_builder),
            "d_reward": float(r_destroyer),
            "val_acc": float(acc) if acc != -1.0 else 0.0,
            "b_loss": float(b_loss),
            "d_loss": float(d_loss),
            "grad_noise": d_res['config']['grad_noise'],
            "hp": b_res['config']
        })

    unique_successful_arch = len(successful_hp_set)
    search_statistics = {
        "total_trials": stats['total_trials'],
        "success_trials": stats['success_trials'],
        "failed_trials": stats['failed_trials'],
        "unique_successful_configs": unique_successful_arch,
        "time": time.time()-time1
    }

    print("========== NAS Statistics ==========")
    print(f"Successful trials : {stats['success_trials']}")
    print(f"Failed trials : {stats['failed_trials']}")
    print(f"Unique successful architectures: {unique_successful_arch}")
    print("====================================")
    return history, search_statistics
