import numpy as np

import ncMaxPipeline as ncm
from pymxs import runtime as rt

path = 'D:/projects/max/unreal_to_biped/Rig_UniformMale_only_main_bones.max'
ncm.open_file(path)

char = ncm.get_fbx_character('root')

bp = ncm.biped()
bp.make()
# bp.v = False
bp.match_to_fbx_character(char)


# bp = ncm.biped()
# # # # fbx_bone = ncm.dummy('thigh_r')
# biped_bone = bp.bones['Bip001 R Calf']
# 
# rot = rt.EulerToQuat(rt.EulerAngles(0, 0, 0))
# rt.biped.setTransform(biped_bone.node, rt.Name('rotation'), rot, False)
# # 
# # 
# biped_bone.rz -= 12.237947637880364
# # (quat 0.669917 0.113105 -0.731091 0.0626383)
# # (quat 0.669917 0.113105 -0.731091 0.0626383)
# print(biped_bone.world_r)


# 11 (quat 0.678157 0.0410522 -0.733603 -0.015648)
# 11 (quat 0.678157 0.0410522 -0.733603 -0.015648)