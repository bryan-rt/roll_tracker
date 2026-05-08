"""GT filename to source-video frame index resolver.

Maps label filenames (frame_NNNNNN.txt) to source video frame numbers,
respecting per-camera annotated ranges and strides from the model manifest.

# TODO TB-EVAL-1: implement frame resolver using manifest schema
"""
