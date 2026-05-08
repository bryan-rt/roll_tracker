"""IoU computation and Hungarian one-to-one matcher.

Vectorized IoU in pixel-space (x1y1x2y2). Hungarian matching via
scipy.optimize.linear_sum_assignment with configurable IoU threshold.

# TODO TB-EVAL-1: implement IoU + Hungarian matcher
"""
