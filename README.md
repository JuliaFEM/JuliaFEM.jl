# JuliaFEM.jl - an open source solver for both industrial and academia usage

[![logo](https://raw.githubusercontent.com/JuliaFEM/JuliaFEM.jl/master/docs/logo/JuliaFEMLogo_256x256.png)](https://github.com/JuliaFEM/JuliaFEM.jl)

[![DOI](https://zenodo.org/badge/35573493.svg)](https://zenodo.org/badge/latestdoi/35573493)
[![License](https://img.shields.io/github/license/JuliaFEM/JuliaFEM.jl.svg)](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/JuliaFEM/JuliaFEM.jl)
[![Build Status](https://travis-ci.org/JuliaFEM/JuliaFEM.jl.svg?branch=master)](https://travis-ci.org/JuliaFEM/JuliaFEM.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaFEM/JuliaFEM.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaFEM/JuliaFEM.jl?branch=master)
[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafem.github.io/JuliaFEM.jl/stable)
[![Latest documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliafem.github.io/JuliaFEM.jl/latest)
[![Issues](https://img.shields.io/github/issues/JuliaFEM/JuliaFEM.jl.svg)](https://github.com/JuliaFEM/JuliaFEM.jl/issues)

JuliaFEM organization web-page: [http://www.juliafem.org](http://www.juliafem.org)

The JuliaFEM project develops open-source software for reliable, scalable,
distributed Finite Element Method.

The JuliaFEM software library is a framework that allows for the distributed
processing of large Finite Element Models across clusters of computers using
simple programming models. It is designed to scale up from single servers to
thousands of machines, each offering local computation and storage. The basic
design principle is: everything is nonlinear. All physics models are nonlinear
from which the linearization are made as a special cases.

At the moment, users can perform the following analyses with JuliaFEM: elasticity,
thermal, eigenvalue, contact mechanics, and quasi-static solutions. Typical examples
in industrial applications include non-linear solid mechanics, contact mechanics,
finite strains, and fluid structure interaction problems. For visualization,
JuliaFEM uses ParaView which prefers XDMF file format using XML to store light
data and HDF to store large data-sets, which is more or less the open-source standard.

## Vision

On one hand, the vision of the JuliaFEM includes the opportunity for massive
parallelization using multiple computers with MPI and threading as well as cloud
computing resources in Amazon, Azure and Google Cloud services together with a
company internal server. And on the other hand, the real application complexity
including the simulation model complexity as well as geometric complexity. Not
to forget that the reuse of the existing material models as well as the whole
simulation models are considered crucial features of the JuliaFEM package. 

Recreating the wheel again is definitely not anybody's goal, and thus we try
to use and embrace good practices and formats as much as possible. We have
implemented Abaqus / CalculiX input-file format support and maybe will in the
future extend to other FEM solver formats. Using modern development environments
encourages the user towards fast development time and high productivity. For
developing and creating new ideas and tutorials, we have used Jupyter notebooks
to make easy-to-use handouts.

The user interface for JuliaFEM is Jupyter Notebook, and Julia language itself
is a real programming language. This makes it possible to use JuliaFEM as a part
of a bigger solution cycle, including for example data mining, automatic geometry
modifications, mesh generation, solution, and post-processing and enabling
efficient optimization loops.

## Installing JuliaFEM

Inside Julia REPL, type:
```julia
Pkg.add("JuliaFEM")
```

## Initial road map

JuliaFEM current status: **project planning**

| Version | Number of degree of freedom | Number of cores |
| ------: | --------------------------: | --------------: |
|   0.1.0 |                   1 000 000 |              10 |
|   0.2.0 |                  10 000 000 |             100 |
|   1.0.0 |                 100 000 000 |           1 000 |
|   2.0.0 |               1 000 000 000 |          10 000 |
|   3.0.0 |              10 000 000 000 |         100 000 |

We strongly believe in the test driven development as well as building on top
of previous work. Thus all the new code in this project should be 100% tested.
Also other people have wisdom in style as well:

[The Zen of Python](https://www.python.org/dev/peps/pep-0020/):

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Errors should never pass silently.
```

## Citing

If you like using our package, please consider citing our [article](https://rakenteidenmekaniikka.journal.fi/article/view/64224/26397)
```
@article{frondelius2017juliafem,
  title={JuliaFEM - open source solver for both industrial and academia usage},
  volume={50}, 
  url={https://rakenteidenmekaniikka.journal.fi/article/view/64224},
  DOI={10.23998/rm.64224},
  number={3},
  journal={Rakenteiden Mekaniikka},
  author={Frondelius, Tero and Aho, Jukka},
  year={2017},
  pages={229-233}
}
```


## Contributing

Developing JuliaFEM encourages good practices, starting from unit testing both
for smaller and larger functions and continuing to full integration testing of
different platforms. 

Interested in participating? Please start by reading
[contributing](http://www.juliafem.org/contributing).
