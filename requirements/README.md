# requirements

This directory contains static lists of _known working_ external dependencies for historical versions of `chemprop`.

## Usage

You can install any version of `chemprop` by simply running `pip install -r X.Y.Z_requirements.txt`.

Note that you can also run `pip install chemprop==X.Y.Z` to get that installed version - this will install more recent versions of the external dependencies (which will probably be more correct and fast) but may have backwards compatibility issues.

## Updating

Run `. get_requirements.sh` with Docker installed and running and all of the requirements files will be re-generated.
This only needs to be done whenever a new tagged release of `chemprop` is created and manually pushed to Docker.
