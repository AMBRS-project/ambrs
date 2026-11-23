# Compatibility and numerical changes

Modal parameters now receive distinct Latin-hypercube coordinates, using
cumulative per-mode widths and placing fractions after number/GMD/GSD.
Earlier coordinates overlapped when affected parameters varied. Audit saved
specifications, seeds and resulting design matrices, then regenerate affected
ensembles whose intended independent varying inputs shared coordinates.
Reassess their sensitivity estimates and conclusions. Constant inputs and
normalized compositions do not justify declaring every historical ensemble
invalid or every final fraction perfectly rank-correlated.
