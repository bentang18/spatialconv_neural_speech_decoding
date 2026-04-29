"""DEPRECATED: within-parcel Perceiver summarizer (pre-amendment ``#26``).

The 2026-04-16 late B-1 amendment retired the within-parcel Perceiver
summarizer. The new Phase-1 spatial path is per-electrode tokens with:

- Stage 2 — whole-grid Conv2d (``grid_mixer.py``, planned)
- Stage 3 — soft parcel embedding via raw ``support[N_e, 15]``
  (``parcel_embedding.py``, planned)

Reasons for retirement (full record in
``docs/v14_core_contract_amendment_2026-04-16.md``):

1. Vertex-collision audit: 23.6% of LH electrodes collide on the same
   fsaverage pial vertex; the within-parcel summarizer cannot tell
   collided electrodes apart in the spatial path.
2. uECoG vs sEEG compression calculus: pooling 22-30 dense electrodes
   per active parcel into one token discards sub-parcel somatotopy at
   exactly the scale the dense device was meant to capture.

Do not import from this module. Use ``grid_mixer`` and
``parcel_embedding`` instead. The file is kept (rather than deleted)
only because the original ``#26`` reasoning chain is referenced from
multiple historical docs.
"""

raise ImportError(
    "speech_decoding.v14.local_summarizer is deprecated by the 2026-04-16 "
    "B-1 amendment. The within-parcel Perceiver summarizer is replaced by "
    "Stage 2 GridMixer + Stage 3 SoftParcelEmbedding. See "
    "docs/v14_core_contract_amendment_2026-04-16.md."
)
