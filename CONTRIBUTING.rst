========================
Contributing to JuliaFEM
========================

Here are the basic steps for contributing to JuliaFEM:

1) Create an account or sign in to `GitHub <https://github.com/>`_.

2) Go to `Git home page <http://git-scm.com/>`_ and download the Git installer.
   Run the installer to get Git on your computer. It is a version control system used
   by GitHub. To learn its basics, go through this
   `Git tutorial <https://try.github.io/levels/1/challenges/1>`_.

3) Install Julia (v0.4+) to your computer. At
   `Julia readme <https://github.com/JuliaLang/julia/blob/master/README.md>`_
   you'll find complete instructions for installing it for your platform.

4) Go to the `JuliaFEM GitHub page <https://github.com/JuliaFEM/JuliaFEM.jl>`_.
   At the top-right corner, press the ``Fork``-button to fork your own copy of
   JuliaFEM to your repository.

5) Clone JuliaFEM from your repository to your computer. Navigate to the folder
   you want to clone it to, and type the following command (inserting your GitHub
   username to its place): ``git clone https://github.com/your_github_username/JuliaFEM.jl.git``

6) You can now navigate to JuliaFEM in the folder you chose at step 5. There
   you'll find the same contents as you see in your GitHub JuliaFEM repository.
   Now, locate the file you want to modify, open it with your desired text
   editor, make the changes and save the new version. If you type ``git status``,
   you'll see that the files you've created or modified are listed under ``untracked files``.

7) Add the files you want to update to the staging area by typing
   ``git add <file1> <file2>...``. If you type ``git status``, you'll see that
   the files added to the staging area are listed under ``Changes to be committed``.
   This process also supports wildcard symbols. If you want to add all the files
   to the staging area, just type ``git add .``. If you want to remove a file
   from the staging area, type ``git reset <file>``.

8) To store the staged files, commit the files to your repository and add a
   description message by typing ``git commit -m "your_message_here"``. The
   message should describe the changes that were made.

9) When you are happy with the commits and want to update them to your
   repository, type ``git push origin master``.

10) Go to your GitHub JuliaFEM repository. You'll notice that the commit you
    have made and pushed is now visible above the JuliaFEM file branch. If you
    click the ``latest commit`` link, you can see the changes made to the file.
    Finally, click ``Pull request`` to create a pull request of the commits
    you've made, so that other contributors can review it.

11) If other contributors ask you to make changes to your pull request, just
    repeat steps 6-9. Your commits will be updated to your original pull request.
    Do this until everyone is satisfied and your pull request can be merged to
    the master branch.

There's also some GUI apps to use git if you don't feel command line comfortable.
For OSX and Windows a good application is `SourceTree <https://www.sourcetreeapp.com>`_,
for Linux, maybe `SmartGit <http://www.syntevo.com/smartgit/>`_ will work.

Developing on local machine
---------------------------

To set up ready for development, git clone it to your development directory and
make symbolic link to julia package directory:

.. code-block:: bash

   cd ~/dev
   git clone https://github.com/JuliaFEM/JuliaFEM.jl
   cd ~/.julia/v0.4
   ln -s ~/dev/JuliaFEM

Editors: Juno is bundled with Julia. Another option is to use vim after installing
`vim support for Julia <https://github.com/JuliaLang/julia-vim>`_.

Testing is made easy by using our `Makefile`. From there one founds convenient
functions `make test`, `make test_file` and `make test_function` to make testing
more rapid.

Use of UTF-8 characters in program code
---------------------------------------
We have decided not to use them. See issue
`#18 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/18>`_.

Supported Julia versions
------------------------
We support Julia versions 0.4+. See issue
`#26 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/26>`_.

Only pull requests to src folder
--------------------------------
See issue `#29 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/29>`_. This
ensures peer review check for contributors and hopefully will decrease the
number of merge conflicts. Before making the pull request run all test: either
type ``julia> Pkg.test("JuliaFEM")`` at REPL or ``julia test/runtests.jl`` at
command line. 

New technology should be introduced through notebooks
-----------------------------------------------------
See issue `#12 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/12>`_. Idea is
to introduce new technology as a notebook for the very beginning. Then when it's
get mature the notebook will serve functional test for the matter. All notebooks
will be included as examples to the documentation. 

FactCheck.jl is used to write test for the JuliaFEM.jl package
--------------------------------------------------------------
See issue `#27 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/27>`_. Use
`FactCheck.jl` package to write the tests. We believe Test Driven Development
thus 100 % test coverage is expected. 

JuliaFEM.jl is using Logging.jl
-------------------------------
See issue `#25 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/25>`_. We have
written a test to check all sources in src folder to find any print statements.
Use Logging.jl instead of println().

Code indentation
----------------
We use 4 spaces like in Python. See issue
`#5 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/5>`_.

Function docstrings
-------------------
We use numpy documentation style in our functions. See
`guide <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
See issue `#5 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/5>`_.

Documentation
-------------
We use restructured text to document this project. Information how to write rst
format is described `here <http://sphinx-doc.org/rest.html>`_. See issue
`#49 <https://github.com/JuliaFEM/JuliaFEM.jl/issues/49>`_.

