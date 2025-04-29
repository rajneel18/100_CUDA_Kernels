# Contrastive Loss in CUDA C

## Overview

Contrastive loss is used in tasks such as metric learning, where the objective is to learn embeddings such that similar items are close and dissimilar ones are far apart.

Given a batch of paired inputs `(x1, x2)` and labels:
- `label = 1` for positive (similar) pairs.
- `label = 0` for negative (dissimilar) pairs.

The contrastive loss is defined as:

For each pair (x1, x2), with label `y`:

- Let `D = ||x1 - x2||_2`
- Then the loss is:

