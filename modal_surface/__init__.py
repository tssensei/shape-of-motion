"""VGGT-carrier latent 3D modal field tools.

The current path builds N-view observation graphs by projecting fixed VGGT
carrier points into stabilized modal-analysis views. Each solved latent file
stores one frequency/mode field ``phi_k`` and one per-view, per-mode complex
offset slice ``alpha[:, k]`` with view 0 fixed as the gauge reference.
"""
