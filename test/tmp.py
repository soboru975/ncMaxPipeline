import ncMaxPipeline as ncm
from pymxs import runtime as rt

ncm.unload_packages()
bp = ncm.biped()
fbx_bone = bp.bones['Bip001 R Finger1']
# fbx_bone.rx += 180 
fbx_bone.rz += 180 