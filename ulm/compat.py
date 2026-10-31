"""Fallbacks for notebook-only helpers when stages run as plain scripts."""
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)
