JuliaDoc
========
JuliaDoc is a Python package providing Sphinx extensions and a theme for the Julia language documentation at https://julia.readthedocs.org/. It can also be used by Julia packages to create documentation that is visually unified with the language documentation.

Usage with ReadTheDocs
----------------------
In the ReadTheDocs admin page for your package's documentation:

1. Turn on the "Use virtualenv" option.

1. Under "Requirements file", enter `doc/requirements.txt`.

1. Add [this `requirements.txt` file](https://gist.github.com/pao/5658342/raw/requirements.txt) to your package repository's `doc` folder. This will tell ReadTheDocs where to find the JuliaDoc theme and extensions.

1. In your `doc.conf.py` file, do at least the following:

```Python
import juliadoc

extensions = ['juliadoc.julia', 'juliadoc.jlhelp']
html_theme_path = [juliadoc.get_theme_dir()]
html_sidebars = juliadoc.default_sidebars()
```

License
-------
The MIT License (MIT)

Copyright (c) 2013 Jeff Bezanson, Stefan Karpinski, Viral Shah, Alan Edelman, et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


Acknowledgements
----------------
The packaging is directly inspired by the [Caktus theme for Sphinx](https://github.com/caktus/caktus-sphinx-theme/).
