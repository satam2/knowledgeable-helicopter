from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from taxiout.artifacts import sha256, write_json
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, align, unique_ids


def serialized_values(prediction):
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite prediction")
    rounded = np.rint(values)
    limits = np.iinfo(np.int32)
    if ((rounded < limits.min) | (rounded > limits.max)).any():
        raise ValueError("int32 prediction overflow")
    return rounded.astype(np.int32)


def validate_submission(file, template, expected=None):
    saved = pq.read_table(file)
    original = pq.read_table(template)
    if saved.schema.remove_metadata() != original.schema.remove_metadata():
        raise ValueError("Submission schema differs from actual template")
    frame, reference = saved.to_pandas(), original.to_pandas()
    if list(frame) != [ID, TARGET]:
        raise ValueError("Submission must contain exactly two required columns")
    unique_ids(frame)
    unique_ids(reference)
    if not np.array_equal(frame[ID], reference[ID]):
        raise ValueError("Submission IDs/order differ from template")
    if not np.isfinite(frame[TARGET]).all():
        raise ValueError("Submission has missing/nonfinite predictions")
    if expected is not None and not np.array_equal(frame[TARGET], expected):
        raise ValueError("Parquet roundtrip changed serialized predictions")
    return {"passed": True, "rows": len(frame), "sha256": sha256(file), "schema": str(saved.schema.remove_metadata()),
            "minimum_sec": int(frame[TARGET].min()), "maximum_sec": int(frame[TARGET].max()), "rounding": "numpy.rint, nearest even; no clipping"}


def build_submission(template, predictions, output):
    template, output = external_path(template), external_path(output)
    schema = pq.read_schema(template).remove_metadata()
    if schema.names != [ID, TARGET] or schema.field(ID).type != pa.float64() or schema.field(TARGET).type != pa.int32():
        raise ValueError("Template schema changed; review serialization policy")
    original = pd.read_parquet(template)
    joined = align(original[[ID]], predictions, ["prediction_sec"])
    values = serialized_values(joined["prediction_sec"])
    table = pa.Table.from_arrays([pa.array(joined[ID], type=schema.field(ID).type), pa.array(values, type=pa.int32())], schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    record = validate_submission(temporary, template, values)
    temporary.replace(output)
    record["template_sha256"] = sha256(template)
    write_json(output.with_suffix(".validation.json"), record)
    return record
