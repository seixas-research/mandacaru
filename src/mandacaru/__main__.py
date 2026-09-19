# -*- coding: utf-8 -*-
# file: __main__.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``python -m mandacaru`` -- the same entry point as the ``mandacaru`` script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
