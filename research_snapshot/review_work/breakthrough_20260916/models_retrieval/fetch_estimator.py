"""Final public estimator and weight-license details; no weight download."""
import fetch_sources as fetch
fetch.OUT=fetch.ROOT/"output/breakthrough_20260916/models_retrieval/estimator/sources"
fetch.SOURCES={
 "tabdpt_estimator":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/src/tabdpt/estimator.py",
 "tabdpt_weights_license":"https://huggingface.co/Layer6/TabDPT/raw/main/LICENSE",
 "tabdpt_weights_config":"https://huggingface.co/Layer6/TabDPT/raw/main/config.json",
 "tabdpt_utils":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/src/tabdpt/utils.py",
 "tabdpt_v13_release":"https://api.github.com/repos/layer6ai-labs/TabDPT-inference/releases/tags/v1.3.0",
 "pytabkit_test_ci":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/.github/workflows/testing.yml",
}
if __name__=="__main__":fetch.main()
