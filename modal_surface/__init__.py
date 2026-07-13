"""Foreground-Gaussian latent 3D modal field tools.

The current path builds N-view pixel-candidate observation graphs directly on
foreground 3DGS Gaussian centers. Each solved latent file stores one
frequency/mode field ``phi_k`` and one per-view, per-mode complex offset slice
``alpha[:, k]`` with view 0 fixed as the gauge reference.
"""
