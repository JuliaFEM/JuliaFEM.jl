import os
import sys
import re

import juliadoc

extensions = ['sphinx.ext.mathjax',
              'juliadoc.julia',
              'juliadoc.jldoctest',
              'juliadoc.jlhelp']

master_doc = 'index'

html_theme_path = [juliadoc.get_theme_dir()]
html_sidebars = juliadoc.default_sidebars()
