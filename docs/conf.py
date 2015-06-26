import juliadoc
import os

extensions = ['juliadoc.julia', 'juliadoc.jlhelp']
html_theme_path = [juliadoc.get_theme_dir()]
html_sidebars = juliadoc.default_sidebars()

print(os.getcwd())
os.system("julia build.jl")
