# JuliaFEM 
Build Status: [![Build Status](https://api.travis-ci.org/ovainola/JuliaFEM.svg?branch=master)](https://travis-ci.org/ovainola/JuliaFEM)

Code Coverage: [![Coverage Status](https://coveralls.io/repos/ovainola/JuliaFEM.svg?branch=master)](https://coveralls.io/r/ovainola/JuliaFEM?branch=master)

The JuliaFEM project develops open-source software for reliable, scalable, distributed Finite Element Methdod.

The JuliaFEM software library is a framework that allows for the distributed processing of large Finite Element Models across clusters of computers using simple programming models. It is designed to scale up from single servers to thousands of machines, each offering local computation and storage. The basic design principle is: everything is nonlinear. All physics models are nonlinear from which the linearization are made as a special cases. 

JuliaFEM current status: project planning

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

Please start by reading some practical examples of [the Julia style.](http://julia.readthedocs.org/en/latest/manual/style-guide/)

Contributors: see [contributors](https://github.com/ovainola/JuliaFEM/blob/master/contributors)
