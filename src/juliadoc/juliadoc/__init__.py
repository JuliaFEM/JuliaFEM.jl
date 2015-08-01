import os

def get_theme_dir():
    """
    Returns path to directory containing this package's theme.
    
    This is designed to be used when setting the ``html_theme_path``
    option within Sphinx's ``conf.py`` file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "theme"))

def default_sidebars():
    """
    Returns a dictionary mapping for the templates used to render the
    sidebar on the index page and sub-pages.
    """
    return {
        '**': ['localtoc.html', 'relations.html', 'searchbox.html'],
        'index': ['searchbox.html'],
        'search': [],
    }
