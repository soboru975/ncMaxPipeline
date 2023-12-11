import ncMaxPipeline as ncm
from pymxs import runtime as rt

ncm.unload_packages()

# ncm.set_auto_frame_range_for_animation_keys()
char = ncm.fbx_character('root')
char.delete_sub_bones()
# for bone in char.bones:
#     bone.axis_tripod = False
#     bone.axis_tripod = True
# 
