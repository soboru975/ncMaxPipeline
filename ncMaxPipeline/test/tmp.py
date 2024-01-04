import ncMaxPipeline as ncm
from pymxs import runtime as rt

ncm.unload_packages()
curve = ncm.curve('straight_line_vec')

print(curve)
print(ncm.Color.YELLOW.value)
curve.color = rt.Color(255, 255, 0)