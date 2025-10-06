\# PTB-XL Federated Learning (DSP5100)



This project implements a federated learning pipeline using the PTB-XL ECG dataset.



\## Structure

\- `data/` — metadata CSVs and optional WFDB ECG files

\- `src/` — all Python code (data loader, models, client/server, etc.)

\- `results/` — saved figures and metrics



\## Steps

1\. Create a virtual environment and install requirements:

&nbsp;  ```bash

&nbsp;  python -m venv .venv

&nbsp;  .venv\\Scripts\\activate

&nbsp;  pip install -r requirements.txt

