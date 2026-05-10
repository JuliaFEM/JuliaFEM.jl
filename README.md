# JuliaFEM.jl

[![logo](https://raw.githubusercontent.com/JuliaFEM/JuliaFEM.jl/master/docs/logo/JuliaFEMLogo_256x256.png)](https://github.com/JuliaFEM/JuliaFEM.jl)

[![DOI](https://zenodo.org/badge/35573493.svg)](https://zenodo.org/badge/latestdoi/35573493)
[![License](https://img.shields.io/github/license/JuliaFEM/JuliaFEM.jl.svg)](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md)

JuliaFEM.jl is an open-source finite element framework written in Julia.
The package is still **0.x**; the repository is in the middle of a deliberate
architectural reset toward a **stable 1.0** with a type-stable, zero-allocation,
GPU-friendly assembly pipeline. Many older READMEs and tutorials still describe
the previous API; when in doubt, trust the code and [`AGENTS.md`](AGENTS.md).
Contributions (bug reports, docs, tests, and features) are welcome.

## Status

Current focus areas:

- `Element{K, P, S, N}` template with compile-time DOF layout.
- `DOFHandler` and `DOFBasedCOOAssembler` (zero-allocation hot paths).
- Microkernel-style physics in `src/domains/{continuum, heat, thermo_elastic}/`.
- Matrix-free `apply_K!`, `apply_M!`, Dirichlet, multipoint constraints,
  IC(0) / Jacobi / block-Jacobi preconditioners and a generalized
  eigensolver in `src/assemblers/`.
- A KernelAbstractions backend for the matrix-free path with a Float32
  Metal smoke test in `test/backend/metal/`.

The legacy element-based API (`Problem`, `update!`, `Analysis`, …) is
still present under `src/legacy/` for backward compatibility but is not
the recommended entry point.

## Installing

```julia
using Pkg
Pkg.add("JuliaFEM")
```

## A modern minimal example

```julia
using JuliaFEM

mesh = create_structured_box_mesh(Hex8;
    xmin = 0.0, xmax = 1.0, nx = 4,
    ymin = 0.0, ymax = 1.0, ny = 4,
    zmin = 0.0, zmax = 1.0, nz = 4,
)

S  = @DOFSet{u::DOF{Displacement{3}, Vertex}}
ET = Element{Hex8, Lagrange{1}, S}
elements, handler = create_elements!(mesh, ET)

material = LinearElastic(E = 210e9, ν = 0.3)
kernel   = ContinuumKernel(ContinuumFormulation{FullThreeD}(),
                           material, Displacement{3}())

asm   = DOFBasedCOOAssembler()
cache = create_cache(asm, elements, handler, mesh, kernel)
assemble!(cache, asm, kernel, mesh)
K, f  = extract_system(cache)
```

The same block is parsed from this file in `test/docs/readme_example.jl`, so it stays
copy-pasteable as the API evolves.

For a matrix-free Krylov solve, see
`test/assemblers/test_dof_based_apply_K.jl` and
`test/assemblers/test_eigensolve.jl`.

## Documentation

- [`docs/repository_layout.md`](docs/repository_layout.md): where to put new files
  (full text under [`docs/src/repository_layout.md`](docs/src/repository_layout.md)).
- [`docs/`](docs/): Documenter-built API reference and user-facing index.
  Historical Jupyter tutorials under [`docs/tutorials/`](docs/tutorials/)
  (2015-2016 API; reference only).
- `src/<topic>/README.md`: short module notes (topology, mesh, materials, ...).

## Contributing

Please read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) before opening a pull
request: fork and branch, keep changes review-sized, run tests, and describe
what you changed.

**Git commits.** Keep history easy to read: small commits, usually one file;
**two files** in one commit is fine when they are inseparable (e.g. a helper and
its only caller). With `.githooks/` and `core.hooksPath`, **pre-commit** caps
staged paths at two and **commit-msg** rejects log lines longer than **80
characters** (wrap subject, summary, and bullets). If that workflow feels
unfamiliar, open your PR with tests passing and ask for help splitting history
in review.

**Code expectations.** Assembly hot paths must stay type-stable and allocation-free
after warmup; CI and [`test/assemblers/test_dof_based_zero_alloc.jl`](test/assemblers/test_dof_based_zero_alloc.jl)
guard that. The full suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Use that locally before opening a PR; CI runs the same tests.

## Citing

If you use JuliaFEM.jl in academic work, please cite

```text
@article{frondelius2017juliafem,
  title   = {Julia{FEM} - open source solver for both industrial and academia usage},
  volume  = {50},
  url     = {https://rakenteidenmekaniikka.journal.fi/article/view/64224},
  doi     = {10.23998/rm.64224},
  number  = {3},
  journal = {Rakenteiden Mekaniikka},
  author  = {Frondelius, Tero and Aho, Jukka},
  year    = {2017},
  pages   = {229-233}
}
```
