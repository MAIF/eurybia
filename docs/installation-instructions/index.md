# Installation instructions

Eurybia supports Python 3.10 and newer. Install it from PyPI:

```bash
pip install eurybia
```

## Jupyter

To display interactive graphs in Jupyter Notebook or JupyterLab, install `ipywidgets` in the environment used by the notebook kernel:

```bash
pip install ipywidgets
```

Recent Jupyter Notebook and JupyterLab releases configure widget support automatically. If the Jupyter server and its kernel use different environments, install `ipywidgets` in the kernel environment as well.

### Check widget support

Run this code in a notebook cell:

```python
from ipywidgets import interact

def f(x):
    return x

interact(f, x=10)
```

### Check Plotly support

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(y=[2, 1, 4, 3]))
fig.add_trace(go.Bar(y=[1, 4, 3, 2]))
fig.update_layout(title="Hello Figure")
fig.show()
```

## Compatibility issues

When using Eurybia, compatibility issues may arise from packages already installed in the environment. Start with a fresh virtual environment and let `pip` resolve Eurybia's declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install eurybia
```
