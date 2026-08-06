# Poster Copy

## Recovering 3D Vibration Modes from Multi-View Videos with Gaussian Splatting

**One sweep video builds the scene. Fixed cameras observe vibration. A shared 3D Gaussian representation lets us lift selected image-space frequencies into spatial 3D motion fields.**

## Why is this hard?

- Optical flow measures horizontal and vertical pixel motion, not full 3D displacement.
- Motion along a camera ray is weakly constrained from one view.
- Thin leaves, masks, and occlusion make direct observations incomplete.
- Independently reconstructed cameras cannot contribute to one shared 3D solution.

## Build one shared static scene

We jointly register sampled sweep frames and one reference image from every fixed camera with COLMAP. The sweep and fixed cameras therefore share one coordinate system. Registered sweep images train separate foreground and background Gaussian sets; only the foreground receives vibration modes.

A Gaussian stores a center, covariance, opacity, and color:

\[
\mathcal G_g=(\mu_g,\Sigma_g,o_g,c_g).
\]

Projected Gaussians are blended from front to back with

\[
w_{p,i}=T_{p,i}a_{p,i},\qquad
C(p)=\sum_i w_{p,i}c_i.
\]

**Plain-language view:** 3DGS represents the scene with many colored, translucent 3D ellipsoids. Their centers provide explicit locations to which we can attach 3D displacement vectors.

## Extract vibration evidence

For every fixed view, foreground masks restrict dense optical flow to the Bush. A Fourier transform converts each pixel trajectory into complex amplitude and phase:

\[
\widehat d_p(f)=\sum_t d_p(t)e^{-i2\pi ft}.
\]

Greedy frequency selection chooses a compact set that explains motion distributed across the foreground. These 2D Fourier fields are measurements, not yet 3D modes.

## Connect pixels to Gaussians

Rendered foreground depth unprojects a valid pixel into the canonical scene. A local KNN search finds nearby foreground Gaussians, which are scored by opacity-weighted Gaussian density:

\[
s_{p,g}=o_g\exp\!\left[-\tfrac12(X_p-\mu_g)^\top
\Sigma_g^{-1}(X_p-\mu_g)\right].
\]

The resulting observation topology records which Gaussian candidates should be constrained by each measured pixel. This is distinct from ordinary alpha compositing, whose purpose is appearance rendering.

## Solve local 3D motion

Mutual 3D KNN edges are removed when Gaussian color or rendered depth changes sharply. The remaining connected sets become local rigid components rather than one globally rigid Bush.

For component \(c\), each Gaussian follows a small complex translation and rotation,

\[
\phi_{k,g}=B_{c,g}\xi_{k,c},
\]

and the component twist is fitted by projecting it into every supporting camera:

\[
\xi_{k,c}^{\star}=
\arg\min_{\xi}\sum_{(v,p,g)\in\mathcal O_c}
\bar w_{v,p,g}
\left\|\alpha_{v,k}J_{v,g}B_{c,g}\xi-
\widehat d_{v,k,p}\right\|_2^2.
\]

Multiple views reduce depth ambiguity; rank and conditioning decide which components become trusted motion seeds.

## Complete the spatial mode

A separate foreground KNN graph propagates motion from trusted seeds while preserving solved values and limiting graph distance. Each selected frequency finally produces

\[
\phi_k\in\mathbb C^{G\times3},
\]

one complex 3D displacement vector per foreground Gaussian.

## Takeaway

The Bush pipeline combines shared camera geometry, frequency-domain optical flow, explicit cross-dimensional correspondence, local structural constraints, and motion fill to recover interpretable spatial 3D vibration patterns. These fields support qualitative inspection and later animation or physical modeling, but are not yet claimed to be physically normalized eigenmodes.
