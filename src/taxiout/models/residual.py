import numpy as np


def proxy_status(proxy, config):
    proxy = np.asarray(proxy, dtype=float)
    return np.where(~np.isfinite(proxy), "missing", np.where((proxy < config["proxy_min"]) | (proxy > config["proxy_max"]), "invalid", "present"))


def residual_target(y, proxy):
    return np.asarray(y, dtype=float) - np.asarray(proxy, dtype=float)


def combine(proxy, correction, fallback, status):
    usable = np.asarray(status) == "present"
    prediction = np.asarray(fallback, dtype=float).copy()
    prediction[usable] = np.asarray(proxy)[usable] + np.asarray(correction)[usable]
    return prediction, np.where(usable, "residual", np.char.add("direct_", np.asarray(status, dtype=str)))
