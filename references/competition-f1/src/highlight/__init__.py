"""AIC 通用视频高光剪辑 — highlight clip selection + per-frame re-framing.

Package layout::

    geometry.py   crop-box geometry (the organizer's max-box definition)
    io/           index / submission / video metadata IO
    metric/       local evaluator + submission validator
    features/     frozen feature extraction
    models/       temporal (highlight) and spatial (crop placement) heads
    decide/       expected-F1 decision layer
    infer/        end-to-end inference over the test set

Nothing here imports ``verl``; this package runs in the lightweight baseline
environment (opencv / numpy / torch), not the 8-GPU RL training environment.
"""

__version__ = "0.1.0"
