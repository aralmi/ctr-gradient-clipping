from collections import deque
import math

import numpy as np
import torch


GRADIENT_METHODS = {
    "none", "global_norm", "value", "agc", "agc_listing", "adagc", "adagc_listing",
    "autoclip", "zclip", "zclip_listing", "softadaclip", "softadaclip_layer",
}


def norm(tensors):
    norms = [torch.linalg.vector_norm(tensor) for tensor in tensors]
    return torch.linalg.vector_norm(torch.stack(norms)).item() if norms else 0.0


def scale_for(raw, threshold):
    return min(1.0, threshold / raw) if raw else 1.0


class GradientClipper:
    def __init__(self, method, **options):
        if method not in GRADIENT_METHODS:
            raise ValueError(f"Unknown gradient clipping method: {method}")
        self.method = method
        self.options = options
        self.history = deque(maxlen=options.get("history", 1000))
        self.steps = 0
        self.local = {}
        self.mean = self.second = 0.0

    def state_dict(self):
        return {
            "steps": self.steps, "local": self.local.copy(), "history": list(self.history),
            "mean": self.mean, "second": self.second,
        }

    def load_state_dict(self, state):
        self.steps = state["steps"]
        self.local = state["local"].copy()
        self.history.clear()
        self.history.extend(state["history"])
        self.mean = state["mean"]
        self.second = state["second"]

    def global_scale(self, grads, factor):
        if factor < 1:
            for _, _, grad in grads:
                grad.mul_(factor)
        return factor < 1

    def history_scale(self, raw):
        if self.method == "autoclip":
            self.history.append(raw)
            if self.steps < self.options.get("warmup", 100):
                return 1.0
            limit = np.percentile(self.history, self.options.get("percentile", 10))
            return scale_for(raw, limit)
        listing = self.method == "zclip_listing"
        alpha = self.options.get("alpha", 0.99 if listing else 0.97)
        k = self.options.get("z", 2.5)
        warmup = self.options.get("warmup", 1 if listing else 25)
        sigma = math.sqrt(max(0.0, self.second - self.mean ** 2))
        z = (raw - self.mean) / (sigma + self.options.get("eps", 1e-6))
        limit = raw
        if self.steps > warmup and self.steps > 1 and z > k:
            limit = self.mean + sigma * (k if listing else k * k / z)
        if self.steps == 1:
            self.mean, self.second = limit, limit ** 2
        else:
            self.mean = alpha * self.mean + (1 - alpha) * limit
            self.second = alpha * self.second + (1 - alpha) * limit ** 2
        return scale_for(raw, limit)

    def local_scale(self, grads, raw):
        changed = False
        warmup = self.steps <= self.options.get("warmup", 100)
        if self.method == "adagc" and warmup:
            factor = scale_for(raw, self.options.get("absolute", 1.0))
            changed = self.global_scale(grads, factor)
            for i, _, grad in grads:
                length = torch.linalg.vector_norm(grad).item()
                if length:
                    self.local[i] = min(self.local.get(i, length), length)
            return changed

        for i, parameter, grad in grads:
            length = torch.linalg.vector_norm(grad).item()
            if length == 0:
                continue
            previous = self.local.get(i, length)
            if self.method in ("agc", "agc_listing"):
                dims = tuple(range(1, parameter.ndim))
                if self.method == "agc_listing" or not dims:
                    dims = None
                weights = torch.linalg.vector_norm(parameter, dim=dims, keepdim=True)
                sizes = torch.linalg.vector_norm(grad, dim=dims, keepdim=True)
                limit = weights.clamp_min(self.options.get("eps", 1e-3))
                limit = limit * self.options.get("factor", 0.01)
                factor = torch.where(sizes > 0, (limit / sizes).clamp(max=1), 1.0)
                grad.mul_(factor)
                changed |= bool((factor < 1).any())
                continue
            if self.method == "adagc":
                beta = self.options.get("beta", 0.99)
                factor = scale_for(length, self.options.get("relative", 1.04) * previous)
                self.local[i] = beta * previous + (1 - beta) * length * factor
            elif self.method == "adagc_listing":
                beta = self.options.get("beta", 0.95)
                self.local[i] = beta * previous + (1 - beta) * length
                weight = max(torch.linalg.vector_norm(parameter).item(), 1e-3)
                factor = scale_for(self.local[i], self.options.get("factor", 0.02) * weight)
            else:
                beta = self.options.get("beta", 0.99)
                self.local[i] = beta * previous + (1 - beta) * length
                ratio = length / max(self.options.get("scale", 1.0) * self.local[i], 1e-8)
                factor = math.tanh(ratio) / ratio
            grad.mul_(factor)
            changed |= factor < 1
        return changed

    @torch.no_grad()
    def clip(self, parameters):
        grads = [(i, p, p.grad) for i, p in enumerate(parameters) if p.grad is not None]
        raw = norm(grad for _, _, grad in grads)
        if not math.isfinite(raw):
            raise FloatingPointError("Non-finite gradient norm")
        if not grads:
            return raw, raw, False

        self.steps += 1
        threshold = self.options.get("threshold", 1.0)
        if self.method == "none":
            return raw, raw, False
        if self.method == "value":
            changed = any(bool((grad.abs() > threshold).any()) for _, _, grad in grads)
            if changed:
                for _, _, grad in grads:
                    grad.clamp_(-threshold, threshold)
        elif self.method == "global_norm":
            changed = self.global_scale(grads, scale_for(raw, threshold))
        elif self.method in ("autoclip", "zclip", "zclip_listing"):
            changed = self.global_scale(grads, self.history_scale(raw))
        elif self.method == "softadaclip":
            factor = math.tanh(threshold / (raw + 1e-8))
            changed = self.global_scale(grads, factor) if raw else False
        else:
            changed = self.local_scale(grads, raw)
        return raw, norm(grad for _, _, grad in grads) if changed else raw, changed


OPTIMIZER_METHODS = {
    "adam", "adamclip", "adamclip_grad", "adamwclip", "stableadamw", "stableadamw_listing",
    "spam", "stable_spam",
}


class ClippedAdam(torch.optim.Optimizer):
    def __init__(self, params, method, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, **options):
        if method not in OPTIMIZER_METHODS:
            raise ValueError(f"Unknown optimizer: {method}")
        defaults = dict(
            method=method, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            iteration=0, **options,
        )
        super().__init__(params, defaults)
        self.last_stats = {}

    def prepare_gradient(self, grad, state, group, reset):
        method = group["method"]
        variance = state["exp_avg_sq"]
        age = state["age"]
        if method == "adamclip_grad" and age > 0:
            bound = group.get("threshold", 1.0) * variance.sqrt() + group["eps"]
            result = grad.clamp(-bound, bound)
            return result, bool((grad.abs() > bound).any())
        if method == "spam" and not reset and age >= group.get("spike_warmup", 10):
            bound = (group.get("spike_threshold", 5000.0) * variance).sqrt()
            spikes = (variance > 0) & (grad.abs() > bound)
            return torch.where(spikes, grad.sign() * bound, grad), bool(spikes.any())
        if method != "stable_spam":
            return grad, False

        t = state["step"]
        eps = group["eps"]
        bmax = group.get("beta_max", 0.999)
        b1 = group.get("beta_norm1", 0.7)
        b2 = group.get("beta_norm2", 0.9)
        maximum = grad.abs().max()
        state["max_ema"] = bmax * state.get("max_ema", 0.0) + (1 - bmax) * maximum
        bound = state["max_ema"] / (1 - bmax ** t)
        spikes = grad.abs() > bound
        grad = torch.where(spikes, grad * (bound / maximum.clamp_min(eps)), grad)
        length = torch.linalg.vector_norm(grad)
        state["norm_m"] = b1 * state.get("norm_m", 0.0) + (1 - b1) * length
        state["norm_v"] = b2 * state.get("norm_v", 0.0) + (1 - b2) * length.square()
        target = (state["norm_m"] / (1 - b1 ** t)) / (
            (state["norm_v"] / (1 - b2 ** t)).sqrt() + eps
        )
        return grad * (target / length.clamp_min(eps)), bool(spikes.any())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grads = [p.grad for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not math.isfinite(norm(grads)):
            raise FloatingPointError("Non-finite gradient norm")
        triggered = False
        effective_gradients = []
        reset_triggered = False
        for group in self.param_groups:
            group["iteration"] += 1
            t = group["iteration"]
            method = group["method"]
            is_spam = method in ("spam", "stable_spam")
            period = group.get("reset_period", 500 if method == "spam" else 2000)
            reset = is_spam and t > 1 and (t - 1) % period == 0
            reset_triggered |= reset
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            threshold = group.get("threshold", 1.0)
            for p in group["params"]:
                state = self.state[p]
                if reset and state:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
                    state["age"] = 0
                if p.grad is None:
                    continue
                if not state:
                    state.update(step=0, age=0, exp_avg=torch.zeros_like(p),
                                 exp_avg_sq=torch.zeros_like(p))
                state["step"] += 1
                gradient = p.grad.add(p, alpha=wd) if method in ("adam", "adamclip") and wd else p.grad
                gradient, clipped = self.prepare_gradient(gradient, state, group, reset)
                triggered |= clipped
                effective_gradients.append(gradient)
                state["age"] += 1
                age = state["age"]
                moment = state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance = state["exp_avg_sq"].mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2,
                )
                corrected_variance = variance / (1 - beta2 ** age)
                update = (moment / (1 - beta1 ** age)) / (corrected_variance.sqrt() + eps)
                lr = group["lr"]
                if method in ("adamclip", "adamwclip", "stableadamw_listing"):
                    bound = threshold
                    if method == "stableadamw_listing":
                        bound *= gradient.square().mean().sqrt().item()
                    triggered |= bool((update.abs() > bound).any())
                    update.clamp_(-bound, bound)
                elif method == "stableadamw":
                    rms = (gradient.square() / corrected_variance.clamp_min(eps ** 2)).mean().sqrt().item()
                    scale = max(1.0, rms / threshold)
                    triggered |= scale > 1
                    lr /= scale
                elif method == "spam":
                    warmup = group.get("reset_warmup", 20)
                    position = (t - 1) % period + 1
                    if t > period and position < warmup:
                        lr *= (1 - math.cos(math.pi * position / warmup)) / 2
                if method not in ("adam", "adamclip"):
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        self.last_stats = {
            "moment_input_norm": norm(effective_gradients), "triggered": triggered,
            "moment_reset": reset_triggered,
        }
        return loss
