#!/bin/bash

version=$1

# Replace in __init__.py
sed -i -e "s/\(__version__ = \).*/\1\"$1\"/" src/astropalmerio/__init__.py
rm src/astropalmerio/__init__.py-e
