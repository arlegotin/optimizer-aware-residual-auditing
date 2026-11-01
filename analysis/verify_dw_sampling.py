# SPDX-License-Identifier: AGPL-3.0-only
"""Replay the published dW-sampling metadata without Torch or corpus access."""
import runpy
from pathlib import Path


def main():
    replay = (Path(__file__).resolve().parent / 'dw_sampling' /
              '20261031T120154Z-reopened' / 'closeout' / 'attempt-0001' /
              'reproduce-recorded-grid.py')
    runpy.run_path(str(replay), run_name='__main__')


if __name__ == '__main__':
    main()
