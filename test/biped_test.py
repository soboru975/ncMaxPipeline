import ncMaxPipeline as ncm
from pymxs import runtime as rt

ncm.unload_packages()

# path = 'D:/projects/max/unreal_to_biped/Rig_UniformMale_only_main_bones.max'
path = 'D:/projects/max/unreal_to_biped/AS_G5M1_hit_Body_R.FBX'
ncm.open_file(path)

# char = ncm.fbx_character('root')
bp = ncm.biped()
bp.make()
# bp.v = False
bp.match_to_fbx_character()
bp.transfer_animation()
# bp.link_to_fbx_character()


