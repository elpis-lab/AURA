"""Shared object dimensions for the cracker-box pushing task."""

import numpy as np

# Mesh AABB from simulation/assets/cracker_box_flipped/textured.obj, same convention used by
# simulation/collect_push_data.py via geometry.object_model.get_obj_shape.
CRACKER_BOX_FLIPPED_SHAPE = np.array([0.163599, 0.213804, 0.071683], dtype=float)
