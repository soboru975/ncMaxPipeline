import ncMaxPipeline as ncm
from pymxs import runtime as rt

ncm.unload_packages()
ncm.open_file()

biped = ncm.Biped()
biped.make()
clavicle = biped.bones['Bip001 L Clavicle']
print(clavicle.world_t)
# clavicle.drt = 'l'
# print(clavicle.world_t)
# 
# clavicle.drt = 'r'
# print(clavicle.world_t)



# # ncm.set_auto_frame_range_for_animation_keys()
# # char = ncm.fbx_character('root')
# # char.delete_sub_bones()
# # for bone in char.bones:
# #     bone.axis_tripod = False
# #     bone.axis_tripod = True
# # 
# 
# ncm.install_project_m_menu()
# # ncm.open_start_up_script_folder()
# 
