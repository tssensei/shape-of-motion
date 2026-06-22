"""Surface-based latent 3D modal field tools.

The package implements the current two-view prototype:

    make-packet
        Convert one view's depth/mask/modal_analysis into sampled 3D surface
        points with attached 2D complex modal observations.

    match-two-views
        Project view1 surface points into view2 and keep only co-visible points
        with consistent mask/depth visibility.

    optimize-two-view
        Fit a shared latent 3D complex modal displacement phi on the co-visible
        surface points so both views' 2D complex modes are explained.
"""
