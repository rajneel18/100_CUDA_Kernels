
# Triplet Loss in CUDA C

## Overview

Triplet loss is commonly used in metric learning to ensure that an anchor is closer to a positive (similar) sample than a negative (dissimilar) one by a margin.

Given:
- Anchor vector \( a \)
- Positive vector \( p \)
- Negative vector \( n \)
- Margin \( \alpha \)

The loss for each triplet is:

\[
L = \max(0, ||a - p||^2 - ||a - n||^2 + \alpha)
\]

Where:
- \( ||a - p||^2 \): Squared distance between anchor and positive
- \( ||a - n||^2 \): Squared distance between anchor and negative
- \( \alpha \): Margin that separates positives from negatives

