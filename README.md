![JuliaFEMLogo](https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/geometry/JuliaFEMLogo_256x256.png) 
# JuliaFEM 

[![Join the chat at https://gitter.im/JuliaFEM/JuliaFEM](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/JuliaFEM/JuliaFEM?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Build Status: [![Build Status](https://travis-ci.org/JuliaFEM/JuliaFEM.jl.svg?branch=master)](https://travis-ci.org/JuliaFEM/JuliaFEM.jl)

Code Coverage: [![Coverage Status](http://coveralls.io/repos/JuliaFEM/JuliaFEM.jl/badge.svg?branch=master)](https://coveralls.io/r/JuliaFEM/JuliaFEM.jl?branch=master)

Documentation: http://www.juliaFEM.org

The JuliaFEM project develops open-source software for reliable, scalable, distributed Finite Element Method.

The JuliaFEM software library is a framework that allows for the distributed processing of large Finite Element Models across clusters of computers using simple programming models. It is designed to scale up from single servers to thousands of machines, each offering local computation and storage. The basic design principle is: everything is nonlinear. All physics models are nonlinear from which the linearization are made as a special cases. 

JuliaFEM current status: project planning

Initial road map for JuliaFEM:

version | number of degree of freedom | number of cores
----------|-----------------------------------------|----------------------
1.0 | 100 000 000 | 1 000
2.0 | 1 000 000 000 | 10 000
3.0 | 10 000 000 000 | 100 000

We strongly believe in the test driven development as well as building on top of previous work. Thus all the new code in this project should be 100% tested. Also other people have wisdom in style as well:

[The Zen of Python](https://www.python.org/dev/peps/pep-0020/)
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

Interested in participating? Please start by reading  [CONTRIBUTING.md](https://github.com/JuliaFEM/JuliaFEM/blob/master/CONTRIBUTING.md).

Contributors: see [contributors](https://github.com/JuliaFEM/JuliaFEM/blob/master/contributors)
