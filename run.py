import numpy as np
from scipy.optimize import optimize
import ncMaxPipeline as ncm
from pymxs import runtime as rt

path = 'D:/projects/max/unreal_to_biped/Rig_UniformMale_only_main_bones.max'
ncm.open_file(path)

char = ncm.get_fbx_character('root')
char.delete_sub_bones()

bp = ncm.biped()
bp.make()
# bp.v = False
bp.match_to_fbx_character(char)


# bp = ncm.biped()
# bone = bp.bones['Bip001 Neck']
# fbx_bone = ncm.dummy('neck_01')
# 
# bone.world_t = fbx_bone.world_t
# bone.world_r = fbx_bone.world_r
