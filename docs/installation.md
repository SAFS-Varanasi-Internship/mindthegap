# Installation

Install from source:

```bash
git clone https://github.com/SAFS-Varanasi-Internship/mindthegap.git
cd mindthegap
pip install .
```

To build the documentation locally:

```bash
pip install ".[docs]"
mkdocs serve
```

Then open <http://127.0.0.1:8000>.

## Cloud banks

Synthetic-cloud banks (`.nc` files) are **not** tracked in git or shipped in the
wheel. They are distributed as GitHub Release assets and downloaded on first use
by `mindthegap.cloud_bank`. See `scripts/upload_cloud_banks.sh` for publishing
new banks.
