"""Tier B training: a residual decision head on top of a frozen backbone's label logits.

Data flows through one record shape (`moelar.train.data.Record`) regardless of source.
See `src/moelar/train/README.md` for the recipe and the reasoning.
"""
