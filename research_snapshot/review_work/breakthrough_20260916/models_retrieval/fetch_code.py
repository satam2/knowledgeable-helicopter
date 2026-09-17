"""Inspect public maintained implementation details without importing them."""
import fetch_sources as fetch
fetch.OUT=fetch.ROOT/"output/breakthrough_20260916/models_retrieval/code/sources"
fetch.SOURCES={
 "pytabkit_tabr":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/nn_models/tabr.py",
 "pytabkit_tabr_interface":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/alg_interfaces/tabr_interface.py",
 "pytabkit_defaults":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/sklearn/default_params.py",
 "pytabkit_nn_interface":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/alg_interfaces/nn_interfaces.py",
 "pytabkit_sklearn":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/sklearn/sklearn_interfaces.py",
 "pytabkit_docs":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/docs/source/models/01_sklearn_interfaces.rst",
 "tabdpt_regressor":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/src/tabdpt/regressor.py",
 "tabdpt_model":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/src/tabdpt/model.py",
 "tabdpt_base":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/src/tabdpt/base.py",
 "tabdpt_example":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/examples/reg_example.py",
 "tabdpt_model_files":"https://huggingface.co/api/models/Layer6/TabDPT",
 "faiss_cpu_package":"https://pypi.org/pypi/faiss-cpu/1.12.0/json",
 "tabr_scaling":"https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/bin/tabr_scaling.py",
}
if __name__=="__main__":fetch.main()
