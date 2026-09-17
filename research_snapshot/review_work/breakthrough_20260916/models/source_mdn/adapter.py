"""Student-t source mixture; analytical mean selection on original raw MSE."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import StudentT
import math
import sys
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import tabm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from encoders import FrameEncoder

CLOCKS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]
COMPONENTS = 6
DEGREES_OF_FREEDOM = 4.
MAX_EPOCHS = 64
PATIENCE = 8
BATCH_SIZE = 4096


def raw_anchors(x):
    values = x[["takeoff_minus_" + clock for clock in CLOCKS]].to_numpy(dtype=np.float64)
    available = np.isfinite(values) & (values != -999999)
    return np.where(available, values, 0.), available


def distribution_parameters(output, anchors, available, minimum_scale):
    if output.ndim != 3 or output.shape[-1] != 18:
        raise ValueError("Mixture network output must have shape rows,members,18")
    if anchors.shape != available.shape or anchors.shape != (output.shape[0], 5):
        raise ValueError("Exactly five aligned observed clock anchors are required")
    allowed = torch.cat([available, torch.ones((len(available), 1), dtype=torch.bool, device=available.device)], dim=1)
    logits = output[:, :, :6]
    log_weights = torch.log_softmax(logits.masked_fill(~allowed[:, None, :], -torch.inf), dim=-1)
    safe_anchors = torch.where(available, anchors, torch.zeros_like(anchors))
    observed = safe_anchors[:, None, :].expand(-1, output.shape[1], -1)
    means = torch.cat([observed + output[:, :, 6:11], output[:, :, 11:12]], dim=-1)
    scales = F.softplus(output[:, :, 12:].double()) + minimum_scale
    return log_weights, means, scales


def analytical_mean(log_weights, means):
    return (log_weights.double().exp() * means.double()).sum(dim=-1)


def member_negative_log_likelihood(parameters, labels):
    log_weights, means, scales = parameters
    distribution = StudentT(df=DEGREES_OF_FREEDOM, loc=means.double(), scale=scales.double())
    component_log_prob = distribution.log_prob(labels.double()[:, None, None])
    return -torch.logsumexp(log_weights.double() + component_log_prob, dim=-1)


class SourceMixture(nn.Module):
    def __init__(self, n_numeric, cardinalities, raw_target_scale):
        super().__init__()
        if not math.isfinite(raw_target_scale) or raw_target_scale < 1.:
            raise ValueError("Affine target scale must be finite and at least one second")
        dimensions = [min(32, max(4, math.ceil(math.sqrt(count)))) for count in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(count, dimension, padding_idx=0)
                                         for count, dimension in zip(cardinalities, dimensions)])
        for embedding in self.embeddings:
            with torch.no_grad():
                embedding.weight[1].zero_()
        self.backbone = tabm.TabM.make(n_num_features=n_numeric + sum(dimensions), d_out=18,
                                       n_blocks=3, d_block=256, k=8, dropout=.1, arch_type="tabm")
        self.register_buffer("minimum_scale", torch.tensor(1. / raw_target_scale, dtype=torch.float64))
        self.register_buffer("gate_bias", torch.tensor([2., 0., 0., 0., -2., 0.]))

    def forward(self, numbers, categories, anchors, available, return_distribution=False):
        features = [numbers] + [embedding(categories[:, i]) for i, embedding in enumerate(self.embeddings)]
        output = self.backbone(torch.cat(features, dim=1))
        output = torch.cat([output[:, :, :6] + self.gate_bias, output[:, :, 6:]], dim=-1)
        parameters = distribution_parameters(output, anchors, available, self.minimum_scale)
        return parameters if return_distribution else analytical_mean(parameters[0], parameters[1])


def tensors(encoder, x, center, scale):
    numbers, categories = encoder.transform(x)
    anchors, available = raw_anchors(x)
    standardized = np.where(available, (anchors - center) / scale, 0.).astype("float32")
    return tuple(torch.from_numpy(values) for values in (numbers, categories, standardized, available))


def infer(network, values, device="cuda", batch_size=8192):
    network.eval()
    result = np.empty(len(values[0]), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(result), batch_size):
            end = min(start + batch_size, len(result))
            member_means = network(*(value[start:end].to(device) for value in values))
            result[start:end] = member_means.mean(dim=1).cpu().numpy()
    return result


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    if not torch.cuda.is_available():
        raise RuntimeError("Source-mixture fit requires a centrally scheduled CUDA slot")
    if steps is not None and (steps < 1 or steps > MAX_EPOCHS or tuning is not None):
        raise ValueError("Refit uses the raw-MSE tune-selected epoch count only")
    if steps is None and tuning is None:
        raise ValueError("Original tune data required for raw-MSE epoch selection")
    y = np.asarray(y, dtype=np.float64)
    if not len(y) or len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Original raw-second labels must be nonempty, finite and aligned")
    if tuning is not None and (len(tuning[0]) != len(tuning[1]) or not np.isfinite(tuning[1]).all()):
        raise ValueError("Original tune raw labels must be finite and aligned")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    encoder = FrameEncoder().fit(x, neural=True)
    center, scale = float(y.mean()), max(float(y.std()), 1.)
    train = tensors(encoder, x, center, scale)
    tune = tensors(encoder, tuning[0], center, scale) if tuning is not None else None
    labels = torch.from_numpy(((y - center) / scale).astype("float64"))
    network = SourceMixture(train[0].shape[1], [len(mapping) + 2 for mapping in encoder.categories.values()], scale).cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    best_mse, best_epoch, best_state = float("inf"), 1, None
    history = []
    epochs = MAX_EPOCHS if steps is None else int(steps)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        network.train()
        total_nll = 0.
        for ids in torch.randperm(len(y), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            parameters = network(*(value[ids].cuda() for value in train), return_distribution=True)
            member_nll = member_negative_log_likelihood(parameters, labels[ids].cuda())
            loss = member_nll.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Source mixture produced nonfinite Student-t negative log likelihood")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 10.)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Source mixture gradients are nonfinite")
            optimizer.step()
            total_nll += float(loss.detach()) * len(ids)
        row = {"epoch": epoch, "train_standardized_nll": total_nll / len(y),
               "train_raw_second_nll": total_nll / len(y) + math.log(scale),
               "seconds": time.monotonic() - epoch_started}
        if tune is not None:
            predicted = infer(network, tune) * scale + center
            if not np.isfinite(predicted).all():
                raise FloatingPointError("Analytical tune means became nonfinite")
            mse = float(np.mean(np.square(predicted - np.asarray(tuning[1], dtype=np.float64))))
            row["tune_raw_mse_sec2"] = mse
            if mse < best_mse:
                best_mse, best_epoch = mse, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
        else:
            best_epoch = epoch
        history.append(row)
        print("SOURCE_MDN", row, flush=True)
        if tune is not None and epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        network.load_state_dict(best_state)
    peak = torch.cuda.max_memory_allocated()
    network.cpu().eval()
    torch.cuda.empty_cache()
    model = {"network": network, "encoder": encoder, "center": center, "scale": scale, "steps": best_epoch, "threads": threads}
    evidence = {"steps": best_epoch, "rows": len(y), "features": len(x.columns), "runtime_sec": time.monotonic() - started,
                "history": history, "gpu_peak_bytes": peak, "torch_version": torch.__version__, "tabm_version": version("tabm"),
                "components": CLOCKS + ["learned_direct_mean"], "degrees_of_freedom": DEGREES_OF_FREEDOM,
                "parameters": {"members": 8, "blocks": 3, "width": 256, "dropout": .1, "batch_size": BATCH_SIZE,
                               "learning_rate": .001, "weight_decay": .0001, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                               "minimum_component_scale_seconds": 1., "gradient_norm_cap": 10., "initial_gate_bias": [2, 0, 0, 0, -2, 0]},
                "loss": "Mean per-member mixture StudentT(df4) negative loglikelihood; torch.distributions with doubleprecision logdensity",
                "prediction": "Analytical weighted component means then memberaverage; affine inverse to original rawseconds",
                "selection": "Best original-tune raw MSE only; not NLL; fresh fullrefit atselected epochcount",
                "target": "Unmodified rawlabels, affine standardization only; no clipping,deletion,logtransform,or sampling",
                "interpretation": "Research prototype: source weights need not represent calibrated physical source probabilities"}
    return model, evidence


def predict(model, x):
    torch.set_num_threads(model["threads"])
    network = model["network"].cuda()
    values = infer(network, tensors(model["encoder"], x, model["center"], model["scale"])) * model["scale"] + model["center"]
    network.cpu()
    torch.cuda.empty_cache()
    if not np.isfinite(values).all():
        raise FloatingPointError("Source-mixture raw mean predictions must remain finite")
    return values
