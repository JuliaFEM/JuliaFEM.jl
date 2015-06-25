===============================
Notes for JuliaFEM contributors
===============================

This very important document need to be done.

For now, read 

https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md


Developing
----------
```bash
cd ~dev/
git clone https://github.com/JuliaFEM/JuliaFEM.jl
cd ~/.julia/v0.4
ln -s ~/dev/JuliaFEM .
```

Use of UTF-8 characters in program code
---------------------------------------
We have decided not to use them. See issue #18.

Supported Julia versions
------------------------
We support Julia versions 0.4+. [See issue #26](https://github.com/JuliaFEM/JuliaFEM.jl/issues/26)

Only pull requests to src folder
--------------------------------
See [issue #29](https://github.com/JuliaFEM/JuliaFEM.jl/issues/29). This ensures peer review check for contributors and hopefully will decrease the number of merge conflicts. Before making the pull request runn all test: either type `julia> Pkg.test("JuliaFEM")` at REPL or `julia test/runtests.jl` at command line. 

New technology should be introduced through notebooks
-----------------------------------------------------
[See issue #12](https://github.com/JuliaFEM/JuliaFEM.jl/issues/12). Idea is to introduce new technology as a notebook for the very beginning. Then when it's get mature the notebook will serve functional test for the matter. All notebooks will be included as examples to the documentation. 

FactCheck.jl is used to write test for the JuliaFEM.jl package
--------------------------------------------------------------
[See issue #27](https://github.com/JuliaFEM/JuliaFEM.jl/issues/27). Use FactCheck.jl package to write the tests. We believe Test Driven Development thus 100 % test coverage is expected. 

JuliaFEM.jl is using Logging.jl
-------------------------------
[See issue #25](https://github.com/JuliaFEM/JuliaFEM.jl/issues/25). We have written a test to check all sources in src folder to find any print statements. Use Logging.jl instead of println(). 
