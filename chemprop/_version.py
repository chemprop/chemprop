__all__ = ['__version__']

# major, minor, patch
version_info = 1, 3, 1

# suffix
suffix = None

# version string
__version__ = '.'.join(map(str, version_info)) + (f'.{suffix}' if suffix else '')
