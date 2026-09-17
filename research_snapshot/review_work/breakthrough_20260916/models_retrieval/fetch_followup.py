"""Corrected official source locations; preserve first retrieval failures."""
import fetch_sources as fetch
fetch.OUT=fetch.ROOT/"output/breakthrough_20260916/models_retrieval/followup/sources"
fetch.SOURCES={
 "tabr_readme":"https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/README.md",
 "tabr_model":"https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/bin/tabr.py",
 "tabr_environment":"https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/environment.yaml",
 "tabr_license":"https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/LICENSE",
 "tabr_tree":"https://api.github.com/repos/yandex-research/tabular-dl-tabr/git/trees/main?recursive=1",
 "pytabkit_tree":"https://api.github.com/repos/dholzmueller/pytabkit/git/trees/main?recursive=1",
 "pytabkit_license":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/LICENSE.txt",
 "realmlp_standalone":"https://raw.githubusercontent.com/dholzmueller/realmlp-td-s_standalone/main/README.md",
 "tabdpt_pyproject":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/pyproject.toml",
 "tabdpt_license":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/LICENSE",
 "tabdpt_package":"https://pypi.org/pypi/tabdpt/json",
 "tabdpt_tree":"https://api.github.com/repos/layer6ai-labs/TabDPT/git/trees/main?recursive=1",
 "tabdpt_weights_card":"https://huggingface.co/Layer6/TabDPT/raw/main/README.md",
}
if __name__=="__main__":fetch.main()
