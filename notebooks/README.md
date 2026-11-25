# Jupyter tutorials

All three notebooks run on CPU with the files in [data](data/). They include saved plots for reading on GitHub. To start a local Jupyter session, run these commands from the repository root and select the environment as your Python kernel.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r notebooks/requirements.txt jupyterlab
jupyter lab notebooks
```

| Notebook | What you do |
| --- | --- |
| [01 · Blind discovery](01_blind_discovery.ipynb) | Score a synthetic ring and disk, reconstruct 12 anonymized GPT-2 feature groups, and reveal a weekday plot on held-out news. |
| [02 · Train a tiny SAE](02_train_a_tiny_sae.ipynb) | Train two small TopK autoencoders on identical generated rows and measure which parts of the planted geometry they preserve. |
| [03 · Calendar steering](03_calendar_steering.ipynb) | Compare an arc with a chord and reproduce the November plot from a compact table of measured Mistral interventions. |

The bundled samples total less than 1 MB. The GPT-2 samples contain numerical SAE codes, decoder directions, and a small held-out weekday point cloud with calendar labels stored separately. The steering table contains a selected set of real intervention records. The training notebook generates its own synthetic dataset. [Dataset details and fields](data/README.md)

[Visual results guide](../RESULTS.md) · [Full experiment and model downloads](../README.md)
