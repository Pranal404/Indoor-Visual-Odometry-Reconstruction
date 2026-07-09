# Reproducibility Notes

The repository is designed to be run with user-supplied RGB-D sequences.

Recommended practice:

- keep raw datasets outside Git;
- copy template configs before editing them;
- record camera intrinsics, depth scale and depth range;
- keep run logs and metrics together with each experiment output;
- do not compare pose metrics as physical accuracy unless an independent
  measured trajectory is available.

The classical metric script reports local motion, path length and trajectory
plots. These are internal consistency checks, not ground-truth pose accuracy.

