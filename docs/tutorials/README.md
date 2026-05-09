# Historical Jupyter tutorials (not the 0.x API)

The notebooks in this directory date from **2015–2016**. They target APIs that
predate the current type-stable pipeline (`Element{K,P,S,N}`, `DOFHandler`,
`AbstractKernel`, modern assemblers).

They are **not** executed by package CI and are **not** built into the Documenter
site. Treat them as archaeology:

- Prefer **`docs/src/index.md`**, **`AGENTS.md`**, and **`test/`** for runnable
  examples.
- For the old `Problem` / Dict-field stack, read **`docs/src/legacy.md`** and
  enable **`JULIAFEM_ENABLE_LEGACY=1`** only if you must reproduce historical
  workflows.

RST index files (`index.rst`, `notebooks.rst`) are legacy Sphinx-era metadata and
may be stale.
