# Present so `mpm_solver_warp` resolves to this directory rather than to the module of the same
# name inside it. Both spellings are in use: the trainer imports `mpm_solver_warp.engine_utils`,
# and the files here import each other flat (`from warp_utils import *`), so the directory and
# its parent are both on the path and the parent has to win.
