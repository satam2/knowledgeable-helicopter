from taxiout.schema import BLOCK, ID, MOVEMENT, PHASE, TARGET, unique_ids, utc

POLICY_VERSION = "dep-hidden-removed-v1-month-isolated"


def make_observations(raw):
    unique_ids(raw)
    if not raw[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unsupported phase")
    dep = raw[PHASE].eq("DEP")
    labels = raw.loc[dep, [ID, TARGET]].copy() if TARGET in raw else None
    # No model uses arrival taxi-in labels, so remove both hidden columns entirely.
    obs = raw.drop(columns=[BLOCK, TARGET], errors="ignore").copy()
    obs[MOVEMENT] = utc(obs[MOVEMENT], required=True)
    codes = obs[MOVEMENT].dt.year * 100 + obs[MOVEMENT].dt.month
    metadata = {"policy_version": POLICY_VERSION, "months": [f"{code // 100:04d}-{code % 100:02d}" for code in sorted(codes.unique())]}
    return obs, labels, metadata


def assert_observations(obs):
    if BLOCK in obs or TARGET in obs:
        raise ValueError("Feature code received hidden departure columns")
    unique_ids(obs)
