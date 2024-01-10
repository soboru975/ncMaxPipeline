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

biped_num = "Bip002"


@dataclass
class _HelperPointPositionPreset:
    bone: str  # Point 포지션 값 정할 때 기준이 되는 본 이름
    position: list  # 본 포지션 값에 add

@dataclass
class _HelperPointPresetBase:
    '''
        헬퍼 포인트들의 초기 포지션 세팅을 위한 정보를 담고 있음
    '''
    Shoulder = _HelperPointPositionPreset('L Clavicle', [10, 0, 5])
    Elbow = _HelperPointPositionPreset('L Forearm', [0, 0, 0])
    Buttock = _HelperPointPositionPreset("L Thigh", [1, 3, 5])
    Knee = _HelperPointPositionPreset("L Calf", [0, 0, 0])

    Spine_Back = _HelperPointPositionPreset("Spine2", [20, 8, 0])
    Spine_BackEnd = _HelperPointPositionPreset("Spine2", [15, 15, 0])
    Spine_Front = _HelperPointPositionPreset("Spine2", [20, -10, 0])
    Spine_FrontEnd = _HelperPointPositionPreset("Spine2", [20, 0, -10])


class _HelperPointsMaker(_HelperPointPresetBase):
    def __init__(self, biped_num):
        self.biped_num = biped_num

    def _get_bone_node(self, bone: str):
        return f"{self.biped_num} {bone}"
        # return nodes._Node(f"{self.biped_num} {bone}")

    def _get_pos(self, bone: str):
        biped = ncm.biped(self.biped_num)
        bone = biped.bones[f'{bone}']
        return bone.world_t

    @property
    def shoulder(self):
        print(self.Shoulder.bone)
        print(self._get_bone_node(self.Shoulder.bone))
        bone_node = self._get_bone_node(self.Shoulder.bone)
        print(self._get_pos(bone_node))
        
    @property
    def elbow(self):
        print(self.Elbow.bone)

    @property
    def buttock(self):
        print(self.Buttock.bone)

    @property
    def knee(self):
        print(self.Knee.bone)
        
        # points_names = []
        # for attr in dir(self):
        #     if isinstance(getattr(type(self), attr), _HelperPointPositionPreset):                
        #         points_names.append(attr)
        # print(points_names)
        # return points_names
    '''
    @helper_points.setter
    def helper_points(self, points_names: list):
        point_full_names = []
        for name in points_names:
            print(name)
            ncm.point(f'{self.biped_num}Point_{name}')
        return point_full_names

    def _set_points(self):
        pass
    '''

# class HelperPoint

test = _HelperPointsMaker('Bip002')
test.shoulder

    
    
'''예시
_HelperBoneMakePresetBase.shoulder.bone
_HelperBoneMakePresetBase.elbow.position[0]

preset = _HelperBoneMakePresetBase

preset.shoulder.bone
'''


'''
@dataclass
class _HelperConstraintSingle:
    target: str
    maintain_offset: bool
    constraint_type: str


@dataclass
class _HelperConstraintAverage:
    targets: List[str]
    maintain_offset: bool
    constraint_type: str

@dataclass
class _HelperPointPRConstraint:
    ''-'
        헬퍼 포인트들의 컨스트레인을 위한 정보를 담고 있음
        로테이션 컨스트레인을 사용하지 않는 포인트도 있어 포지션과 로테이션을 분리함
    ''-'
    shoulder_p = _HelperConstraintSingle('Bip Clavicle', False, "Link")
    shoulder_r = _HelperConstraintSingle('BoneTwist-Lower', False, "LookAt")

    elbow_p = _HelperConstraintSingle("Bip UpperArm", False, "Link")
    elbow_r = _HelperConstraintAverage(["Bone L UpperArm Twist1", "Bone L ForeArm"], False, "Orientation")

    buttock_p = _HelperConstraintSingle("Bip Pelvis", False, "Link")

    knee_p = _HelperConstraintSingle("Bip Thigh", False, "Link")
    knee_r = _HelperConstraintAverage(["Bone L Thigh Twist1", "Bone L Calf Twist"], False, "Orientation")

    spine_back_p = _HelperConstraintSingle("Bip UpperArm", False, "Link")
    spine_back_r = _HelperConstraintSingle("Spine_BackEnd", False, "Link")

    spine_backEnd_p = _HelperConstraintSingle("Bip Spine1", False, "Link")

    spine_front_p = _HelperConstraintAverage(["Bip Spine2", "Bone L UpperArm"], True, "Position")
    spine_front_r = _HelperConstraintSingle("Spine_FrontEnd", False, "LookAt")

    spine_frontEnd_p = _HelperConstraintSingle("Bip Spine1", False, "Link")


@dataclass
class _HelperBonePreset:
    target: str  # constraint target point
    parent: str  # hierarchy

@dataclass
class _HelperBonePresetBase:
    ''-'
        헬퍼본은 헬퍼포인트 값을 취하지만 계층 구조는 바이패드 본에 소속됨
        포지션 = 포지션 컨스트레인
        로테이션 = 오리엔테이션 컨스트레인
    ''-'
    shoulder = _HelperBonePreset('Point L Shoulder', "Bip Clavicle")
    elbow = _HelperConstraintSingle("Point L Elbow", "Bip UpperArm")
    buttock = _HelperConstraintSingle("Point Buttock", "Bip Pelvis")
    knee = _HelperConstraintSingle("Point Spine Knee", "Bip UpperArm")
    spine_back = _HelperConstraintSingle("Point Spine Back", "Bip Clavicle")
    spine_backEnd = _HelperConstraintSingle("Point Spine BackEnd", "Point Spine Back")
    spine_front = _HelperConstraintSingle("Point Spine Front", "Bip UpperArm")
    spine_frontEnd = _HelperConstraintSingle("Point Spine FrontEnd", "Point Spine Front")
    
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
