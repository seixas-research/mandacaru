# -*- coding: utf-8 -*-
# file: __main__.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""``python -m carcara`` -- the same entry point as the ``carcara`` script."""

import sys

from .cli import main

sys.exit(main())
