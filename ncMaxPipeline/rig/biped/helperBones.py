import ncMaxPipeline as ncm
ncm.unload_packages()

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
import pymxs
rt = pymxs.runtime

import ncMaxPipeline as ncm
from ncMaxPipeline.rig import nodes

import ncMaxPipeline.core as cmds
# point_name = ncm.nodes._Object("test")

# LC = ncm.nodes.point("test")
# print(LC.world_t)

@dataclass
class _HelperPointPositionPreset:
    bone: str  # Point 포지션 값 정할 때 기준이 되는 본 이름
    position: list  # 본 포지션 값에 add

    
@dataclass
class _HelperPointPresetBase:
    """
        헬퍼 포인트들의 초기 포지션 세팅을 위한 정보를 담고 있습니다.
    """
    Shoulder = _HelperPointPositionPreset('L Clavicle', [10, 0, 5])
    Elbow = _HelperPointPositionPreset('L Forearm', [0, 0, 0])
    Buttock = _HelperPointPositionPreset("L Thigh", [1, 3, 5])
    Knee = _HelperPointPositionPreset("L Calf", [0, 0, 0])
    
    SpineBack = _HelperPointPositionPreset("Spine2", [20, 8, 0])
    SpineBackEnd = _HelperPointPositionPreset("Spine2", [15, 15, 0])
    SpineFront = _HelperPointPositionPreset("Spine2", [20, -10, 0])
    SpineFrontEnd = _HelperPointPositionPreset("Spine2", [20, 0, -10])

    
class _HelperPointsMakerBase(_HelperPointPresetBase):
    def __init__(self, biped_num):
        self.biped_num = biped_num
        self.name = None
        self.bone = None
        
    def _bone_name(self):
        return f"""{self.biped_num} {self.bone}"""
    
    def _point_name(self):
        return f"""Point Helper L {self.name}"""
    
    @property
    def make_point(self):        
        """
        Notes:
            Point를 생성하는 함수입니다.
            :rtype: str
        """        
        point = ncm.Point()
        point.make(self._point_name())        
        return f"{self._point_name()}"
        
    def _get_bone_pos(self):
        biped = ncm.Biped(self.biped_num)
        bone = biped.bones[self.bone]
        return bone.world_t
    
    def _set_reference_pos(self):
        print("!!!!!!!!!!!!---------------!!!!!!!!!!")
        print(self._bone_name())
        pos = self._get_bone_pos(self._bone_name())
        node = ncm.Node(self.make_point)
        node.world_t = pos


class _HelperPointsMaker(_HelperPointsMakerBase):    
    @classmethod
    @property
    def shoulder(cls):
        cls.name = "Soulder"
        cls.bone = _HelperPointPresetBase.Shoulder.bone
        print(cls.bone)
        cls._set_reference_pos()
        return 'tttt'

    @classmethod
    @property
    def elbow(cls):
        cls.name = "Elbow"
        cls.bone = _HelperPointPresetBase.Elbow.bone
        return cls._set_reference_pos()

    @classmethod
    @property
    def buttock(cls):
        cls.name = "Buttock"
        cls.bone = _HelperPointPresetBase.Buttock.bone
        return cls._set_reference_pos()

    @classmethod
    @property
    def knee(cls):
        cls.name = "Knee"
        cls.bone = _HelperPointPresetBase.Knee.bone
        return cls._set_reference_pos()
    
    def make(self):        
        self.shoulder        
        self.elbow
        self.buttock
        self.knee
    
    def abcd(self):
        print('ㅎㅎㅎㅎ')

    def efg(self):
        print('ㄱㄱㄱㄱ')


# class HelperPoint
test = _HelperPointsMaker('Bip001')
test.shoulder()


'''예시
_HelperBoneMakePresetBase.shoulder.bone
_HelperBoneMakePresetBase.elbow.position[0]

preset = _HelperBoneMakePresetBase

preset.shoulder.bone
'''


'''
def create_Point(biped_name):
    side = "L"

    name_dict = {"L Clavicle": ["Shoulder"], "L UpperArm": [""], "L Forearm": ["Elbow"],
                 "L Thigh": ["Hips"], "L Calf": ["Knee"],
                 "Spine2": ["Spine_Back", "Spine_BackEnd", "Spine_Front", "Spine_FrontEnd"]}

    add_pos = {"Shoulder": [10, 0, 5], "Elbow": [0, 0, 0],
               "Hips": [1, 3, 5], "Knee": [0, 0, 0],
               "Spine_Back": [20, 8, 0], "Spine_BackEnd": [15, 15, 0],
               "Spine_Front": [20, -10, 0], "Spine_FrontEnd": [20, 0, -10]}

    for biped in name_dict.keys():        
        pos = ncm.point(f"{biped_name} {biped}")
        # pos = ncm.xform(f"{biped_name} {biped}", q=1, t=1)
        for lc in name_dict[biped]:
            if lc == "":
                pass
            else:
                lc_pos = [pos[0] + add_pos[lc][0], pos[1] + add_pos[lc][1], pos[2] + add_pos[lc][2]]
                LC = ncm.point(n=f"Point_{side}_{lc}", p=lc_pos)
                print(LC)
                # ncm.Color(LC, side="L")
'''
