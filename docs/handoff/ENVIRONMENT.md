# Research Environment

The project ran on Windows with CPython 3.12.14, an NVIDIA RTX 5080 (16 GiB VRAM),
and approximately 64 GiB host RAM. The recorded driver was 591.86. GPU availability
and package downloads on the teammate's machine have not been verified.

The snapshots in `dependencies/` come from the two working environments:

| File | Role | Torch |
| --- | --- | --- |
| `review.pip-freeze.txt` | Core tests and CPU verification | 2.14.0+cpu |
| `breakthrough.pip-freeze.txt` | Neural/CatBoost GPU research | 2.14.0+cu130 |

Both passed `pip check` during handoff preparation. Shared versions include NumPy
1.26.4, pandas 2.3.3, PyArrow 14.0.2, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10,
scikit-learn 1.7.2, TabM 0.0.3 and rtdl_num_embeddings 0.0.12. Preserve the complete
pins rather than installing only this short list. The original baseline
`requirements.lock.txt` does not describe the entire later research environment.

Create fresh 64-bit Python 3.12 environments. Install the selected snapshot from
trusted package sources, with the matching official PyTorch index where required.
These are installed-version records, not wheel bundles or hash-locked package
downloads; availability of every exact build remains to be checked. A generic
default-index install may not resolve the CPU/CUDA Torch build suffixes.

After installing dependencies, install this package with
`python -m pip install --no-deps --no-build-isolation -e .` to avoid silently
replacing the chosen research dependencies. Run `python -m pip check` and focused
synthetic tests using the explicit environment interpreter.

Windows runners have documented native import-order requirements. Preserve Torch
before LightGBM for the designated GPU paths and the recorded CPU wrapper imports.
The installed LightGBM API supported `eval_X`/`eval_y`; do not assume a different
package build is compatible merely because its family name matches.

Virtual environments referenced a base Python outside the original workspace and
are not portable. macOS cannot run the CUDA paths unchanged. CPU inference for
individual components needs its own replay checks; cross-platform bitwise parity
is not promised. Use `-B` to avoid bytecode changes in frozen source snapshots.

For large fits, use one heavy GPU process, two CPU threads per fitting job, and
explicit host-memory reservations. The latest runner had a 20 GiB process limit,
28 GiB post-import startup requirement and 8 GiB host reserve. These are documented
controls, not evidence that all jobs fit a smaller machine.
