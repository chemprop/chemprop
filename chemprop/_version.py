__all__ = ['__version__']

# major, minor, patch
version_info = 1, 0, 3

# suffix
suffix = 'beta'

# version string
__version__ = '.'.join(map(str, version_info)) + (f'.{suffix}' if suffix else '')
