===============================
Notes for JuliaFEM contributors
===============================

This very important document need to be done.

For now, read 

https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md


Developing
----------

git clone https://github.com/JuliaFEM/JuliaFEM
cd ~/.julia/v0.4
ln -s ~/dev/JuliaFEM .


Use of UTF-8 characters in program code
---------------------------------------
We have decided not to use them. See issue #18.

Supported Julia versions
------------------------
We support Julia versions 0.4+

Only pull requests to src folder
--------------------------------
See [issue #29](https://github.com/JuliaFEM/JuliaFEM/issues/29). This ensures peer review check for contributors and hopefully will decrease the number of merge conflicts. 

New technology should be introduced through notebooks
-----------------------------------------------------
[See issue #12](https://github.com/JuliaFEM/JuliaFEM.jl/issues/12). Idea is to introduce new technology as a notebook for the very beginning. Then when it's get mature the notebook will serve functional test for the matter. All notebooks will be included as examples to the documentation. 
