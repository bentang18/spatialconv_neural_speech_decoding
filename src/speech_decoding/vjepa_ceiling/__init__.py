"""V-JEPA 2 teacher-ceiling probe (video-FM mirror of whisper_ceiling).

Measures how linearly decodable each Neuroprobe task is from a frozen V-JEPA 2
ViT-L encoder, as the upper bound on what a video-teacher distillation target
could transfer. Built to parallel ``whisper_ceiling`` exactly: same probe, same
splits, same per-trial NPZ cache — only the teacher (audio Whisper -> video
V-JEPA 2) and the spatial axis differ.
"""
