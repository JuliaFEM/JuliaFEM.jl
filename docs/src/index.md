# JuliaFEM.jl documentation

```@contents
Pages = ["index.md", "api.md"]
```

The JuliaFEM project develops open-source software for reliable, scalable,
distributed Finite Element Method.

The JuliaFEM software library is a framework that allows for the distributed
processing of large Finite Element Models across clusters of computers using
simple programming models. It is designed to scale up from single servers to
thousands of machines, each offering local computation and storage. The basic
design principle is: everything is nonlinear. All physics models are nonlinear
from which the linearization are made as a special cases. 

## Installing and testing package

Installing package goes same way like other packages in julia, i.e.
```julia
julia> Pkg.add("JuliaFEM")
```

Testing package can be done using `Pkg.test`, i.e.
```julia
julia> Pkg.test("JuliaFEM")
```

## Contributing

Have a new great idea and want to share it with the open source community?
From [here](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md)
and [here](https://juliadocs.github.io/Documenter.jl/stable/man/contributing/)
you can look for coding style. [Here](https://docs.julialang.org/en/stable/manual/packages/#Making-changes-to-an-existing-package-1) is explained how to contribute to
open source project, in general.
