from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Union, List, Optional

import numpy as np
from pymxs import runtime as rt
from pymxs import animate as anim
from pymxs import attime as at

import ncMaxPipeline as ncm
from ncMaxPipeline.rig import nodes


def biped(name: str = 'Bip001'):
    """biped 객체를 반환한다
    
    centimeter를 기준으로 작업이 되어 있어서 이를 바꿔준다.
    """
    if str(rt.units.SystemType) != 'centimeters':
        print(f"System unit is not centimeters: {rt.units.SystemType}")
        print(f"Automatically change system unit to centimeters")
        rt.units.SystemType = rt.Name('centimeters')
    bp = _Biped(name)
    bp.make_preset = _BipedMakePresets.PROJECT_M
    bp.bone_preset = _BipedBonePresets.PROJECT_M
    return bp


def fbx_character(root: str = 'root'):
    """fbx로 불러온 캐릭터의 root bone이름을 넣어준다."""
    return _FBXCharacter(root)


def get_biped_names():
    """바이패드 객체들의 이름을 반환한다"""
    pass


@dataclass
class _BipedMakePresetBase:
    height: float = 170  # 키 cm
    angle: float = -90  # 회전축
    make_arms: bool = True  # 팔을 만들지 여부
    spine_joint_count: int = 3  # 등뼈 갯수
    neck_joint_count: int = 2  # 목뼈 갯수
    finger_joint_count: int = 3  # 손가락뼈 갯수
    finger_count: int = 5  # 손가락 갯수
    metacarpal: bool = True  # 손바닥뼈
    reduce_thumb_count: bool = True  # 엄지 갯수를 줄일지 여부
    toe_joint_count: int = 1  # 발가락뼈 갯수
    triangle_neck: bool = True  # 하이라키 변경 
    # 켜주면 clavicle이 spin아래에 있다.. 꺼주면 clavicle이 neck 아래에 있다.


class _BipedMakeProjectMPreset(_BipedMakePresetBase):
    pass


class _BipedMakePresets:
    PROJECT_M = _BipedMakeProjectMPreset()


@dataclass
class _BipedBoneData:
    fbx_bone: Optional[str]
    mirror: bool = False


@dataclass
class _BipedBonePresetBase:
    """biped의 뼈들을 fbx의 뼈들로 매칭시키기 위한 정보를 담고 있는 클래스
    
    1:1로 매칭이 가능한 fbx의 뼈만 기록한다.
    만약 root같이 대응되는 biped뼈가 없거나
    pelvis같이 계산상으로 위치를 잡아주는 것들은 여기에 기록하지 않는다.
    """
    Com = _BipedBoneData('pelvis')
    Pelvis = _BipedBoneData('pelvis')
    Spine = _BipedBoneData('spine_01')
    Neck = _BipedBoneData('neck_01')
    Neck1 = _BipedBoneData('neck_02')
    Head = _BipedBoneData('head')
    Clavicle = _BipedBoneData('clavicle', True)
    UpperArm = _BipedBoneData('upperarm', True)
    Forearm = _BipedBoneData('lowerarm', True)
    Hand = _BipedBoneData('hand', True)
    Finger0 = _BipedBoneData('thumb_01', True)
    Finger01 = _BipedBoneData('thumb_02', True)
    Finger02 = _BipedBoneData('thumb_03', True)
    Finger1 = _BipedBoneData('index_metacarpal', True)
    Finger11 = _BipedBoneData('index_01', True)
    Finger12 = _BipedBoneData('index_02', True)
    Finger13 = _BipedBoneData('index_03', True)
    Finger2 = _BipedBoneData('middle_metacarpal', True)
    Finger21 = _BipedBoneData('middle_01', True)
    Finger22 = _BipedBoneData('middle_02', True)
    Finger23 = _BipedBoneData('middle_03', True)
    Finger3 = _BipedBoneData('ring_metacarpal', True)
    Finger31 = _BipedBoneData('ring_01', True)
    Finger32 = _BipedBoneData('ring_02', True)
    Finger33 = _BipedBoneData('ring_03', True)
    Finger4 = _BipedBoneData('pinky_metacarpal', True)
    Finger41 = _BipedBoneData('pinky_01', True)
    Finger42 = _BipedBoneData('pinky_02', True)
    Finger43 = _BipedBoneData('pinky_03', True)
    Thigh = _BipedBoneData('thigh', True)
    Calf = _BipedBoneData('calf', True)
    Foot = _BipedBoneData('foot', True)
    Toe0 = _BipedBoneData('ball', True)

    @property
    def bone_names(self):
        names = []
        for attr in dir(self):
            type(getattr(type(self), attr))
            if isinstance(getattr(type(self), attr, None), _BipedBoneData):
                names.append(attr)
        return names


class _BipedBoneProjectMPreset(_BipedBonePresetBase):
    pass


class _BipedBonePresets:
    PROJECT_M = _BipedBoneProjectMPreset()


class _BipedBoneAnimationTransferBase(ABC):
    def __init__(self, bone: '_BipedBone'):
        self.bone = bone

    def __call__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame

        self._make_points()
        self._execute_match()
        self._delete_points()

    @property
    @abstractmethod
    def is_matched(self):
        pass

    @abstractmethod
    def _execute_match(self):
        pass

    @property
    def biped(self):
        return self.bone.biped

    def _make_points(self):
        self.bone_point = ncm.point('bone_pt')
        self.fbx_point = ncm.point('fbx_pt')

    def _delete_points(self):
        self.bone_point.delete()
        self.fbx_point.delete()


class _ComAnimationTransfer(_BipedBoneAnimationTransferBase):

    @property
    def is_matched(self):
        return (self.biped.name == self.bone.name) or \
               (self.bone.pure_name in ['Thigh', 'Calf', 'Foot', 'Toe0', 'Spine', 'Clavicle'])

    def _execute_match(self):
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)

        self.bone_point.node.transform = self.bone.node.transform
        self.fbx_point.node.transform = fbx_bone.node.transform

        self.bone_point.parent = self.fbx_point
        self.fbx_point.parent = fbx_bone.name

        with anim(True):
            for f in range(self.start_frame, self.end_frame):
                with at(f):
                    self.bone.node.transform = self.bone_point.node.transform


class _NotSetYetBoneAnimationTrasfer(_BipedBoneAnimationTransferBase):
    """아직 설정하지 않은 bone들
    """

    @property
    def is_matched(self):
        return True

    def _execute_match(self):
        print(f"Not set animation transfer yet: {self.bone.name}")


class _BipedBoneToFBXBoneMatcherBase(ABC):
    def __init__(self, bone: '_BipedBone'):
        self.bone = bone

    def __call__(self):
        self._execute_match()

    @property
    @abstractmethod
    def is_matched(self):
        pass

    @property
    def biped(self):
        return self.bone.bones.biped

    @abstractmethod
    def _execute_match(self):
        pass

    def _get_fbx_bone_length(self, fbx_bone: nodes._Dummy):
        """fbx 뼈의 길이를 구한다."""
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child_fbx_bone.world_t)
        return bone_length

    def _set_key_to_bone(self):
        """bone을 고정하기 위하여 키를 준다.
    
        Notes:
            분명 회전값을 줬는데 내가 회전을 준대로 고정이 잘 안되어 있다면
            아래의 두가지 키 주는 방식에서 방법을 바꿔보자.
            setKey or setSelectedKey
        """
        if self.bone.is_com:
            rt.biped.addNewKey(self.bone.node.controller.vertical.controller, ncm.get_current_frame())
            rt.biped.addNewKey(self.bone.node.controller.horizontal.controller, ncm.get_current_frame())
            rt.biped.addNewKey(self.bone.node.controller.turning.controller, ncm.get_current_frame())
        else:
            rt.biped.setKey(self.bone.node, False, False, False)
            # if self.bone.pure_name in ['Clavicle', 'UpperArm', 'Forearm', 'Hand', 'Head']:
            #     rt.biped.setKey(self.bone.node, False, False, False)
            # elif any(name in self.bone.pure_name for name in ['Finger', 'Neck']):
            #     rt.biped.setKey(self.bone.node, False, False, False)
            # else:
            #     rt.biped.setSelectedKey(self.bone.node.controller)


class _BipedBoneScaleToFBXBoneMatcherBase(_BipedBoneToFBXBoneMatcherBase):
    pass


class _BipedBonePoseToFBXBoneMatcherBase(_BipedBoneToFBXBoneMatcherBase):
    """scale 을 변경할 때는 반드시 figure mode를 켜야 한다."""

    def __call__(self):
        self._execute_match()
        self._set_key_to_bone()


class _ComScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.biped.name == self.bone.name

    def _execute_match(self):
        """scale은 필요없다"""
        pass


class _ComPoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.biped.name == self.bone.name

    def _execute_match(self):
        """com 위치 설정

        Notes:
            biped의 thigh는 독립적으로 위치조절을 할수가 없다.
                단지 com의 위치만을 따라가는 것이다.
                그러므로 import해온 fbx에 thigh의 biped bone을 맞추려면
                com의 위치를 맞추어야 한다.  
    
                z, y만 맞추고 x는 pelvis에서 scale로 맞춰야 한다...
                지금 fbx가 ue5에서 가져온거라 90도 회전이 되어있는데
                다른 캐릭터는 또 계산해야 하는 축이 다를 수 있겠다. 
                우선 축 문제는 신경 쓰지 않고 작업한다.
            
            figure mode를 키는 이유
                이건 spine 때문이다. spine의 위치 변경은 피규어 모드를 켯을때만 가능하다.
                com의 경우 피규어 모드를 끄고 위치를 이동하면
                피규어 모드를 켰을때 다시 처음 생성했을 때의 com의 위치로 변경되어 버린다.
                그래서 spine의 위치를 똑바로 잡으려면 com이 피규어 모드을때 위치를 잡고 있어야한다. 
        """
        self.biped.figure_mode = True
        self._match_position()
        self._match_rotation()
        self.biped.figure_mode = False

    def _match_rotation(self):
        thigh_l = ncm.dummy('thigh_l')
        thigh_r = ncm.dummy('thigh_r')
        axis_y = (thigh_l.world_t - thigh_r.world_t) / np.linalg.norm(thigh_l.world_t - thigh_r.world_t)

        pelvis = ncm.dummy('pelvis')
        spine_1 = ncm.dummy('spine_01')
        axis_z = (spine_1.world_t - pelvis.world_t) / np.linalg.norm(spine_1.world_t - pelvis.world_t)

        axis_x = np.cross(axis_y, axis_z)

        mat = rt.Matrix3(rt.Point3(*axis_x.tolist()),
                         rt.Point3(*axis_y.tolist()),
                         rt.Point3(*axis_z.tolist()),
                         rt.Point3(0, 0, 0))
        self.bone.world_r = mat.rotation

    def _match_position(self):
        thigh_l = ncm.dummy('thigh_l')
        thigh_r = ncm.dummy('thigh_r')
        self.bone.world_t = (thigh_l.world_t + thigh_r.world_t) / 2


class _ToeBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        """toe 이름은 보통 숫자가 붙어있다.

        ex) Bip001 L Toe0
        """
        return 'Toe' in self.bone.pure_name

    def _execute_match(self):
        fbx_toe_len = self._get_fbx_toe_length()
        foot_scale = rt.biped.getTransform(self.biped.bones[self.bone.parent].node,
                                           rt.Name('scale'))
        old_toe_scale = rt.biped.getTransform(self.bone.node, rt.Name('scale'))
        scale = rt.Point3(old_toe_scale.x * fbx_toe_len / old_toe_scale.x,
                          old_toe_scale.y,
                          foot_scale.z)
        rt.biped.setTransform(self.bone.node, rt.Name('scale'), scale, False)

    def _get_fbx_toe_length(self):
        if self.bone.drt == 'L':
            fbx_toe = 'ball_l'
            fbx_toe_end = 'toeEnd_l'
        else:
            fbx_toe = 'ball_r'
            fbx_toe_end = 'toeEnd_r'

        return float(np.linalg.norm(ncm.dummy(fbx_toe).world_t - ncm.dummy(fbx_toe_end).world_t))


class _ToeBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        """toe 이름은 보통 숫자가 붙어있다.

        ex) Bip001 L Toe0
        """
        return 'Toe' in self.bone.pure_name

    def _execute_match(self):
        """위치는 foot에서 이미 맞추었다"""
        pass


class _FootBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Foot'

    def _execute_match(self):
        """스케일을 수정한다.
        
        Notes:
            toe의 위치를 맞추기 위해서는 toe의 위치를 옮기는게 아니고 foot의 스케일을 조절해야 한다.
            
            길이를 맞추는 방법
                1. fbx foot과 fbx toe와의 위치를 구한다.
                   fbx foot에서 직선을 내려서 직각 삼각형을 구한다.
                2. 그 비율대로 biped foot의 스케일을 조절한다.            
        """
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        fbx_foot_ptn = fbx_bone.world_t
        fbx_toe_ptn = ncm.dummy(fbx_bone.children[0]).world_t
        fbx_heel_ptn = fbx_foot_ptn.copy()
        fbx_heel_ptn[2] = fbx_toe_ptn[2]

        bone_foot_ptn = self.bone.world_t
        bone_toe_ptn = self.biped.bones[self.bone.children[0]].world_t
        bone_heel_ptn = bone_foot_ptn.copy()
        bone_heel_ptn[2] = bone_toe_ptn[2]

        x_rate = np.linalg.norm(fbx_foot_ptn - fbx_heel_ptn) / np.linalg.norm(bone_foot_ptn - bone_heel_ptn)
        y_rate = np.linalg.norm(fbx_heel_ptn - fbx_toe_ptn) / np.linalg.norm(bone_heel_ptn - bone_toe_ptn)

        old_scale = rt.biped.getTransform(self.bone.node, rt.Name('scale'))
        scale = rt.Point3(float(old_scale.x * x_rate),
                          float(old_scale.y * y_rate),
                          old_scale.z)
        rt.biped.setTransform(self.bone.node, rt.Name('scale'), scale, False)


class _FootBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Foot'

    def _execute_match(self):
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        self._match_rotation(fbx_bone)

    def _match_rotation(self, fbx_bone: nodes._Dummy):
        """rotation을 설정한다.

        fbx과의 축을 완전히 일치 시키지 않는다.
        biped의 발바닥면이 xy평면과 일치하게 해달라는 애니메이션팀의 요청에 따라
        rx값만 수정한다.
        """
        ball = ncm.dummy(fbx_bone.children[0])
        angle = ncm.get_angle_between_position_to_axis_on_plane(self.bone.name,
                                                                axis='y', plane='yz',
                                                                target_position=ball.world_t)
        self.bone.rx += angle


class _SpineBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        """3개의 spine을 처음 spine 이름인 spine에서 전부 적용한다."""
        return 'Spine' in self.bone.pure_name

    def _execute_match(self):
        """spine은 3개짜리로 가정하고 제작되었다."""
        biped_spines_bones = self.biped.bones.find_bones_by_pure_name('Spine')
        whole_spine_length = self._get_whole_spine_length()

        self._set_length_to_spines(biped_spines_bones, whole_spine_length)
        self._set_width_to_spines(biped_spines_bones)

    def _set_width_to_spines(self, biped_spines_bones):
        """가로가 너무 얇쌍해서 pelvis의 크기에 맞춘다."""
        pelvis_sz = rt.biped.getTransform(self.biped.bones['Pelvis'].node, rt.Name('scale')).z * 1.5
        old_scale = rt.biped.getTransform(biped_spines_bones[0].node, rt.Name('scale'))
        for bone in biped_spines_bones:
            scale = rt.Point3(old_scale.x, old_scale.y, pelvis_sz)
            rt.biped.setTransform(bone.node, rt.Name('scale'), scale, False)

    def _set_length_to_spines(self, biped_spines_bones: List['_BipedBone'], whole_spine_length: float):
        """spine들의 길이를 설정한다"""
        old_scale = rt.biped.getTransform(biped_spines_bones[0].node, rt.Name('scale'))
        old_spine_length = old_scale.x * 3
        scale_ratio = whole_spine_length / old_spine_length
        for bone in biped_spines_bones:
            scale = rt.Point3(old_scale.x * scale_ratio, old_scale.y, old_scale.z)
            rt.biped.setTransform(bone.node, rt.Name('scale'), scale, False)

    def _get_whole_spine_length(self):
        """spine 전체 길이를 반환한다

        애니메이션팀에서 특정하게 원하는 길이가 디테일하게 없어서
        대략적으로만 계산한다.
        첫번째 spine에서 어깨까지의 직선 길이를 허리 길이로 한다.
        """
        first_fbx_bone_ptn = ncm.dummy('spine_01').world_t
        shoulder_fbx_bones_ptn = (ncm.dummy('clavicle_l').world_t + ncm.dummy('clavicle_r').world_t) / 2
        return float(np.linalg.norm(first_fbx_bone_ptn - shoulder_fbx_bones_ptn))


class _SpineBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        """3개의 spine을 처음 spine 이름인 spine에서 전부 적용한다."""
        return self.bone.pure_name == 'Spine'

    def _execute_match(self):
        """spine은 3개짜리로 가정하고 제작되었다."""
        self.biped.figure_mode = True
        biped_spines_bones = self.biped.bones.find_bones_by_pure_name('Spine')

        self._match_first_bone_position(biped_spines_bones)
        self._set_rotation_rx_to_spines(biped_spines_bones)
        self.biped.figure_mode = False

    def _set_rotation_rx_to_spines(self, biped_spines_bones: List['_BipedBone']):
        """애니메이션팀 요청으로 spine들이 척추를 따라 약간씩 회전하도록 한다."""
        shoulder_fbx_bones_ptn = (ncm.dummy('clavicle_l').world_t + ncm.dummy('clavicle_r').world_t) / 2
        angle_y = ncm.get_angle_between_position_to_axis_on_plane(biped_spines_bones[0].name,
                                                                  axis='x', plane='xz',
                                                                  target_position=shoulder_fbx_bones_ptn)
        angle_z = ncm.get_angle_between_position_to_axis_on_plane(biped_spines_bones[0].name,
                                                                  axis='x', plane='xy',
                                                                  target_position=shoulder_fbx_bones_ptn)
        for bone in biped_spines_bones:
            bone.ry += angle_y / 3
            bone.rz += angle_z / 3

    def _match_first_bone_position(self, biped_spines_bones: List['_BipedBone']):
        """첫 spine의 위치를 맞춘다
        
        Notes:
            biped가 엉망이라서 다음의 순서를 지켜야 한다.
                spine의 경우는 위치를 맞추는게 피규어 모드를 켰을 때만 가능하다.
                그러므로 먼저 피규어 모드를 켜야하지만
                피규어 모드를 키면 또 com의 위치가 이상한 곳으로 튀어 버린다.
                그래서 반드시 com의 위치도 피규어 모드를 키고 변경해야 한다.
                그러면 피규어 모드를 키더라도 com의 위치는 변경이 되지 않는다.
        """
        first_fbx_bone = ncm.dummy('spine_01')
        biped_spines_bones[0].world_t = first_fbx_bone.world_t


class _PelvisBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Pelvis'

    def _execute_match(self):
        """pelvis 크기 설정

        Notes:
            여기서 사이즈 조절을 해주는 것은 오로지 thigh의 위치를 맞추기 위한 
            이유 이다. pelvis를 가로로 늘려서 biped의 thigh의 위치를 맞추는 것이다.
        """
        self._match_size()

    def _match_size(self):
        thigh_l = ncm.dummy('thigh_l')
        thigh_r = ncm.dummy('thigh_r')
        thigh_gap = np.linalg.norm(thigh_l.world_t - thigh_r.world_t)
        self.bone.size = thigh_gap


class _PelvisBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Pelvis'

    def _execute_match(self):
        """pelvis 위치는 com의 위치로 결정한다."""
        pass


class _ClavicleBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Clavicle'

    def _execute_match(self):
        if self.bone.drt == 'L':
            fbx_bone = ncm.dummy('clavicle_l')
        else:
            fbx_bone = ncm.dummy('clavicle_r')
        self.bone.world_t = fbx_bone.world_t
        self._match_to_scale(fbx_bone)

    def _match_to_scale(self, fbx_bone: '_BipedBone'):
        child = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child.world_t)
        self.bone.size = bone_length


class _ClavicleBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Clavicle'

    def _execute_match(self):
        if self.bone.drt == 'L':
            fbx_bone = ncm.dummy('clavicle_l')
        else:
            fbx_bone = ncm.dummy('clavicle_r')
        self._match_to_position(fbx_bone)
        self._match_to_rotation(fbx_bone)

    def _match_to_position(self, fbx_bone: '_BipedBone'):
        """어깨의 위치는 figure mode를 켜야한 가능하다"""
        self.biped.figure_mode = True
        mat = fbx_bone.world_mat
        self.bone.world_t = mat.pos
        self.biped.figure_mode = False

    def _match_to_rotation(self, fbx_bone: '_BipedBone'):
        """회전을 fbx bone과 일치 시킨다."""
        self.bone.world_r = fbx_bone.world_r
        self._finetune_rotation(fbx_bone)

    def _finetune_rotation(self, fbx_bone):
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        dum = ncm.dummy(fbx_bone.name + '_match_guide_dum')
        dum.world_mat = fbx_bone.world_mat
        grp = dum.make_grp()
        if self.bone.drt == 'L':
            angle = ncm.get_angle_between_position_to_axis_on_plane(self.bone.name,
                                                                    axis='x', plane='xy',
                                                                    target_position=child_fbx_bone.world_t)
            dum.rz += angle
            self.bone.world_r = dum.world_r
        else:
            dum.rz += 180
            self.bone.world_r = dum.world_r
            angle = ncm.get_angle_between_position_to_axis_on_plane(self.bone.name,
                                                                    axis='x', plane='xy',
                                                                    target_position=child_fbx_bone.world_t)
            dum.rz += angle
            self.bone.world_r = dum.world_r
        grp.delete()
        dum.delete()


class _UpperArmBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):
    """상박 매칭"""

    @property
    def is_matched(self):
        return self.bone.pure_name in ['UpperArm']

    def _execute_match(self):
        self.bone.size = self._get_fbx_bone_length(ncm.dummy('upperarm_l'))


class _UpperArmBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """"""

    @property
    def is_matched(self):
        return self.bone.pure_name in ['UpperArm']

    def _execute_match(self):
        if self.bone.drt == 'L':
            fbx_bone = ncm.dummy('upperarm_l')
            hand_fbx_bone = ncm.dummy('hand_l')
        else:
            fbx_bone = ncm.dummy('upperarm_r')
            hand_fbx_bone = ncm.dummy('hand_r')
        self.bone.world_r = fbx_bone.world_r
        if self.bone.drt == 'R':
            self.bone.ry += 180
        angle = ncm.get_angle_between_position_to_axis_on_plane(self.bone.name,
                                                                'y', 'yz',
                                                                hand_fbx_bone.world_t)
        self.bone.rx += (angle - 180)


class _ForearmBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Forearm']

    def _execute_match(self):
        self.bone.size = self._get_fbx_bone_length(ncm.dummy('lowerarm_l'))


class _ForearmBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Forearm']

    def _execute_match(self):
        """
        
        Notes:
            설명
                먼저 팔을 쭉 핀다음 (팔꿈치가 일자가 되게)
                fbx bone들의 각도만큼 구부려 준다.
        """
        parent_bone = self.biped.bones[self.bone.parent]
        self.bone.world_r = parent_bone.world_r

        if self.bone.drt == 'L':
            fbx_bone_1 = ncm.dummy('upperarm_l')
            fbx_bone_2 = ncm.dummy('lowerarm_l')
            fbx_bone_3 = ncm.dummy('hand_l')
        else:
            fbx_bone_1 = ncm.dummy('upperarm_r')
            fbx_bone_2 = ncm.dummy('lowerarm_r')
            fbx_bone_3 = ncm.dummy('hand_r')

        angle = ncm.get_angle_between_vectors(fbx_bone_1.world_t - fbx_bone_2.world_t,
                                              fbx_bone_3.world_t - fbx_bone_2.world_t)
        self.bone.rz += (180 - angle)


class _HandBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Hand']

    def _execute_match(self):
        """scale은 필요없다
        
        손의 metacarpal의 경우에는 피규어 모드에서 위치로 세팅하므로 
        여기서 scale로 손가락들의 위치를 맞추는 일은 없다.
        """
        pass


class _HandBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Hand']

    def _execute_match(self):
        """

        Notes:
            설명
                먼저 팔을 쭉 핀다음 (팔꿈치가 일자가 되게)
                fbx bone들의 각도만큼 구부려 준다.
        """
        if self.bone.drt == 'L':
            fbx_bone = ncm.dummy('hand_l')
        else:
            fbx_bone = ncm.dummy('hand_r')
        self.bone.world_r = fbx_bone.world_r
        if self.bone.drt == 'R':
            self.bone.rx += 180
            self.bone.ry += 180


class _MetacarpalBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Finger0',
                                       'Finger1',
                                       'Finger2',
                                       'Finger3',
                                       'Finger4']

    def _execute_match(self):
        """"""
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        self.bone.size = self._get_fbx_bone_length(fbx_bone)


class _MetacarpalBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Finger0',
                                       'Finger1',
                                       'Finger2',
                                       'Finger3',
                                       'Finger4']

    def _execute_match(self):
        """

        Notes:
            설명
                이걸 굳이 local로 변환해서 계산하는 이유는 
                팔의 각도가 figure mode를 끄고 키를 준 상태이기 때문이다.
                
                metacarpal의 경우 figure mode를 켜야만 위치를 변경할 수 있다.
                figure mode를 키면 팔이 엉뚱한 곳으로 가있기 때문에 world 위치를 적용할 수 없다.
        """
        self.biped.figure_mode = True
        if self.bone.drt == 'L':
            hand_fbx_bone = ncm.dummy('hand_l')
            hand_bone = self.biped.bones['Bip001 L Hand']
        else:
            hand_fbx_bone = ncm.dummy('hand_r')
            hand_bone = self.biped.bones['Bip001 R Hand']

        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        local_mat = fbx_bone.world_mat * rt.Inverse(hand_fbx_bone.world_mat)
        if self.bone.drt == 'L':
            mat = local_mat * hand_bone.world_mat
        else:
            rot_mat = rt.Matrix3(1)
            rot_mat.rotation = rt.Eulerangles(0, 0, 180)
            mat = local_mat * (rot_mat * hand_bone.world_mat)

        self.bone.world_t = mat.pos
        self.biped.figure_mode = False
        self.bone.world_r = fbx_bone.world_r
        if self.bone.drt == 'R':
            self.bone.rx += 180
            self.bone.ry += 180


class _FingersBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return 'Finger' in self.bone.pure_name and len(self.bone.pure_name) > 7

    def _execute_match(self):
        """fbx import 할때 손끝이 빠져있는 경우가 있다"""
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        if fbx_bone.children:
            self.bone.size = self._get_fbx_bone_length(fbx_bone)
        else:
            self.bone.size = self._get_fbx_bone_length(ncm.dummy(fbx_bone.parent)) * 0.9


class _FingersBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return 'Finger' in self.bone.pure_name and len(self.bone.pure_name) > 7

    def _execute_match(self):
        """

        Notes:
            설명
                먼저 팔을 쭉 핀다음 (팔꿈치가 일자가 되게)
                fbx bone들의 각도만큼 구부려 준다.
        """
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        self.bone.world_r = fbx_bone.world_r
        if self.bone.drt == 'R':
            self.bone.rx += 180
            self.bone.ry += 180


class _NeckBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return 'Neck' in self.bone.pure_name

    def _execute_match(self):
        """fbx import 할때 손끝이 빠져있는 경우가 있다"""
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        self.bone.size = self._get_fbx_bone_length(fbx_bone)


class _NeckBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return 'Neck' in self.bone.pure_name

    def _execute_match(self):
        """

        Notes:
            위치 이동이 figure mode를 켜야만 가능하다.
        """
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        self.biped.figure_mode = True
        self.bone.world_t = fbx_bone.world_t
        self.biped.figure_mode = False
        self.bone.world_r = fbx_bone.world_r
        self.bone.rx = 180


class _HeadBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Head'

    def _execute_match(self):
        pass


class _HeadBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Head'

    def _execute_match(self):
        """

        Notes:
            위치 이동이 figure mode를 켜야만 가능하다.
        """
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        self.bone.world_r = fbx_bone.world_r
        self.bone.rx = 180


class _ThighBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Thigh']

    def _execute_match(self):
        self._match_biped_bone(self.bone.pure_name.lower())

    def _match_biped_bone(self, fbx_bone_name: str):
        if self.bone.drt == 'L':
            fbx_bone = ncm.dummy(fbx_bone_name + '_l')
        else:
            fbx_bone = ncm.dummy(fbx_bone_name + '_r')
        if not ncm.exists_objects(fbx_bone.name):
            raise ValueError(f"Cannot find fbx bone: {fbx_bone}")
        self._match_size(fbx_bone)

    def _match_size(self, fbx_bone: nodes._Dummy):
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child_fbx_bone.world_t)
        self.bone.size = bone_length


class _ThighBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """허벅지 매칭

    단순하게 fbx 뼈와 matrix 상의 매칭을 한다. 이후 스케일을 조절
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Thigh']

    def _execute_match(self):
        self._match_biped_bone(self.bone.pure_name.lower())

    def _match_biped_bone(self, fbx_bone_name: str):
        if self.bone.drt == 'L':
            fbx_bone = ncm.dummy(fbx_bone_name + '_l')
        else:
            fbx_bone = ncm.dummy(fbx_bone_name + '_r')
        if not ncm.exists_objects(fbx_bone.name):
            raise ValueError(f"Cannot find fbx bone: {fbx_bone}")
        self._match_rotation(fbx_bone)
        self._match_size(fbx_bone)

    def _match_rotation(self, fbx_bone: nodes._Dummy):
        self._match_rotation_to_fbx_bone(fbx_bone)
        self._set_rotation_y_to_foot_bone()

    def _set_rotation_y_to_foot_bone(self):
        """ry값을 조절한다.

        fbx를 제작할때 다리가 1자가 아닌 경우가 있다.
        biped는 무조건 다리가 일자인 것을 전제로만 작업을 할 수 있다.
        그래서 허벅지를 ry방향으로(좌우로) 약간 회전시켜서 foot를 xy평면과 일치시켜야 한다.
        일치 된 후 rz를 돌려주면 정확히 foot의 위치와 동일해 진다.
        """
        if self.bone.drt == 'L':
            calf_bone = self.biped.bones['Bip001 L Calf']
            foot_fbx = ncm.dummy('foot_l')
        else:
            calf_bone = self.biped.bones['Bip001 R Calf']
            foot_fbx = ncm.dummy('foot_r')

        foot_ptn = ncm.get_point3(foot_fbx.world_t)
        local_foot_ptn = foot_ptn * rt.Inverse(self.bone.world_mat)
        local_foot_ptn.y = 0
        world_foot_ptn = local_foot_ptn * self.bone.world_mat

        vec_1 = calf_bone.world_t - self.bone.world_t
        vec_2 = np.array([world_foot_ptn.x, world_foot_ptn.y, world_foot_ptn.z]) - self.bone.world_t
        angle = ncm.get_angle_between_vectors(vec_1, vec_2)
        if self.bone.drt == 'L':
            self.bone.ry += angle
        else:
            self.bone.ry -= angle

    def _match_rotation_to_fbx_bone(self, fbx_bone):
        """fbx 뼈와 일치 시킨다."""
        self.bone.world_r = fbx_bone.world_r
        if self.bone.drt == 'L':
            self.bone.rz = 180

    def _match_size(self, fbx_bone: nodes._Dummy):
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child_fbx_bone.world_t)
        self.bone.size = bone_length


class _CalfBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):
    """종아리 매칭

    크기만 매칭 시킨다
    """

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Calf'

    def _execute_match(self):
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        if not ncm.exists_objects(fbx_bone.name):
            raise ValueError(f"Cannot find fbx bone: {fbx_bone}")
        self._match_size(fbx_bone)

    def _match_size(self, fbx_bone: nodes._Dummy):
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child_fbx_bone.world_t)
        self.bone.size = bone_length


class _CalfBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """종아리 매칭"""

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Calf'

    def _execute_match(self):
        fbx_bone = ncm.dummy(self.bone.preset_fbx_bone_name)
        if not ncm.exists_objects(fbx_bone.name):
            raise ValueError(f"Cannot find fbx bone: {fbx_bone}")
        self._match_rotation()

    def _match_rotation(self):
        # self.bone.node.rotation = fbx_bone.node.rotation
        if self.bone.drt == 'L':
            foot_fbx = ncm.dummy('foot_l')
        else:
            foot_fbx = ncm.dummy('foot_r')
        angle = ncm.get_angle_between_position_to_axis_on_plane(self.bone.name,
                                                                'x', 'xy', foot_fbx.world_t)
        self.bone.rz += angle


class _CommonBoneAttributes:
    bone = None
    biped = None

    def _is_matched(self):
        return self.bone.pure_name in []

    def _get_fbx_bone(self):
        fbx_bone_name = getattr(self.biped.bone_preset, self.bone.pure_name).fbx_bone
        return ncm.dummy(fbx_bone_name + '_l')


class _CommonBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase, _CommonBoneAttributes):
    """일반적인 경우의 뼈들"""

    @property
    def is_matched(self):
        return self._is_matched()

    def _execute_match(self):
        """자식과의 거리로 크기를 맞춘다."""
        fbx_bone = self._get_fbx_bone()
        fbx_bone_len = self._get_fbx_bone_length(fbx_bone)
        self.bone.size = fbx_bone_len


class _CommonBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase, _CommonBoneAttributes):
    """일반적인 뼈 매칭"""

    @property
    def is_matched(self):
        return self._is_matched()

    def _execute_match(self):
        fbx_bone = self._get_fbx_bone()
        self.bone.world_r = fbx_bone.world_r


class _NotSetYetBoneScaleToFBXBoneMatcher(_BipedBoneScaleToFBXBoneMatcherBase):
    """아직 설정하지 않은 bone들
    """

    @property
    def is_matched(self):
        return True

    def _execute_match(self):
        print(f"Not set scale yet: {self.bone.name}")


class _NotSetYetBonePoseToFBXBoneMatcher(_BipedBonePoseToFBXBoneMatcherBase):
    """아직 설정하지 않은 bone들
    """

    def __call__(self):
        self._execute_match()

    def _execute_match(self):
        print(f"Not set pose yet: {self.bone.name}")

    @property
    def is_matched(self):
        return True


class _BipedBoneMirror:
    """이거를 하면 발목만 빼고는 모두 mirror가 잘 된다.
    
    피규어 모드를 키고 꺼가면서 거의 꼼수로 하는 것이긴 하지만.
    하지만 어쨌든 발목의 문제가 계속 해결이 안되서 결국에는 mirror를 하지 못하여 
    양쪽 모두를 직접 조절하는 방식으로 진행하였다. 그래서 이 구문은 백업차원에서 남겨놓는다. 
    """

    def __init__(self, biped: '_Biped'):
        self.biped = biped

    def __call__(self):
        ptn = self.biped.bones[0].world_t
        collection_name = 'left_bones'
        left_bones = self.biped.bones.find_bones_by_direction('L')
        ctrl = left_bones[0].node.controller
        collection = rt.biped.createCopyCollection(ctrl, collection_name)
        pos = rt.biped.copyBipPosture(ctrl, collection, [bone.node for bone in left_bones], rt.Name('snapNone'))
        rt.biped.pasteBipPosture(ctrl, pos, True, rt.Name('pstdefault'), False, False, False, True)
        self.biped.figure_mode = True
        rt.biped.pasteBipPosture(ctrl, pos, True, rt.Name('pstdefault'), False, False, False, True)
        self.biped.figure_mode = False
        rt.biped.pasteBipPosture(ctrl, pos, False, rt.Name('pstdefault'), False, False, False, True)
        self.biped.bones[0].world_t = ptn


class _BipedBone(nodes._Object):
    """biped용 bone 클래스
    
    부모 클래스:
        처음에는 _bone이었으나 max의 bone 객체에 대한 분석이 되지 않아서
        max의 bone과 biped의 bone이 어느정도 서로 연관이 있는지 모른다.
        그래서 우선 _node를 바로 받았다.
        
    Biped bone의 rotation:
        biped는 transform을 결정하는 부모 노드가 없다.
        그래서 local rotation을 움직이면 local rotation이 다시 0으로 바뀐다.
    """

    def __init__(self, bones: '_BipedBones', name: str):
        super(_BipedBone, self).__init__(name)
        self.bones = bones
        self._set_match_guide()
        self._set_match_scale_to_fbx_bone()
        self._set_match_pose_to_fbx_bone()
        self._set_transfer_animation()

    @property
    def preset_fbx_bone_name(self):
        if self.is_com:
            return getattr(self.biped.bone_preset, "Com").fbx_bone
        else:
            if self.pure_name in dir(self.biped.bone_preset):
                name = getattr(self.biped.bone_preset, self.pure_name).fbx_bone
                if self.drt == 'L':
                    name += '_l'
                elif self.drt == 'R':
                    name += '_r'
                return name
            else:
                None

    @property
    def is_com(self):
        return len(self.name.split(' ')) == 1

    @property
    def rx(self):
        return 0

    @rx.setter
    def rx(self, value):
        self.r = (value, 0, 0)

    @property
    def ry(self):
        return 0

    @ry.setter
    def ry(self, value):
        self.r = (0, value, 0)

    @property
    def rz(self):
        return 0

    @rz.setter
    def rz(self, value):
        self.r = (0, 0, value)

    @property
    def r(self):
        rot = rt.QuatToEuler(self.node.transform.rotation)
        return np.array([rot.x, rot.y, rot.z])

    @r.setter
    def r(self, value):
        world_rot_mat = rt.Matrix3(1)
        world_rot_mat.rotation = rt.EulerAngles(float(self.world_r[0]), float(self.world_r[1]), float(self.world_r[2]))
        local_rot_mat = rt.Matrix3(1)
        local_rot_mat.rotation = rt.EulerToQuat(rt.EulerAngles(float(value[0]),
                                                               float(value[1]),
                                                               float(value[2])))
        rt.biped.setTransform(self.node, rt.Name('rotation'), (local_rot_mat * world_rot_mat).rotation, False)

    @property
    def world_t(self):
        ptn = self.node.transform.position
        return np.array([ptn.x, ptn.y, ptn.z])

    @world_t.setter
    def world_t(self, value):
        if isinstance(value, (np.ndarray, tuple, list)):
            ptn = rt.Point3(float(value[0]), float(value[1]), float(value[2]))
            rt.biped.setTransform(self.node, rt.Name('pos'), ptn, False)
        elif isinstance(value, type(rt.Point3)):
            rt.biped.setTransform(self.node, rt.Name('pos'), value, False)
        else:
            raise ValueError(type(value))

    @property
    def world_r(self):
        rot = rt.QuatToEuler(self.node.transform.rotation)
        return np.array([rot.x, rot.y, rot.z])

    @world_r.setter
    def world_r(self, value):
        """biiped의 bone의 rotation을 설정하는 방법은 약간 다르다"""
        if isinstance(value, (np.ndarray, tuple, list)):
            rot = rt.EulerToQuat(rt.EulerAngles(float(value[0]), float(value[1]), float(value[2])))
            rt.biped.setTransform(self.node, rt.Name('rotation'), rot, False)
        elif isinstance(value, type(rt.Rotation)):
            rt.biped.setTransform(self.node, rt.Name('rotation'), value, False)
        else:
            raise ValueError(type(value))

    @property
    def world_mat(self):
        return self.node.transform

    @world_mat.setter
    def world_mat(self, value):
        """scale이 적용이 잘 되려나"""
        rt.biped.setTransform(self.node, rt.Name('pos'), value.position, False)
        rt.biped.setTransform(self.node, rt.Name('rotation'), value.rotation, False)
        rt.biped.setTransform(self.node, rt.Name('scale'), value.scale, False)

    @property
    def size(self):
        return rt.biped.getTransform(self.node, rt.Name('scale'))

    @size.setter
    def size(self, value):
        """biped의 bone의 크기를 설정한다
        
        일반적인 scale과 약간 개념이 다르다. 값 하나로 size를 조절하면 전체 크기가 
        가장 긴쪽을 기준으로 맞춰진다. 설명이 이해가 되려나 모르겠네
        """
        if isinstance(value, (float, np.float64)):
            scale = rt.Point3(float(value), float(value), float(value))
            rt.biped.setTransform(self.node, rt.Name('scale'), scale, False)
        else:
            raise NotImplementedError(value, type(value))

    @property
    def is_mirrored(self):
        splits = self.name.split(' ')
        if len(splits) <= 2:
            return False
        elif len(splits) == 3:
            return True
        else:
            raise NotImplementedError(splits)

    @property
    def drt(self):
        splits = self.name.split(' ')
        if len(splits) == 3:
            return splits[1]
        else:
            return None

    @property
    def pure_name(self):
        """
        Notes:
            예를 들어
            Bip001 L Finger02
            이런 이름이면 biped 이름과 방향을 빼고 순수한 노드의 이름만 반환한다.
            
            네이밍 방식이 바뀌면 코드를 수정해야 할 것이다.    
        """
        splits = self.name.split(' ')
        if len(splits) == 1:
            return splits[0]
        elif len(splits) > 1:
            return splits[-1]
        else:
            raise NotImplementedError

    @property
    def biped(self) -> '_Biped':
        return self.bones.biped

    @property
    def is_available_match_guide(self):
        return self.match_guide.is_available

    def make(self, name: str):
        pass

    def _set_match_scale_to_fbx_bone(self):
        for sub_cls in _BipedBoneScaleToFBXBoneMatcherBase.__subclasses__():
            ins = sub_cls(self)
            if ins.is_matched:
                self.match_scale_to_fbx_bone = ins
                return
        else:
            raise NotImplementedError(f"Cannot find match to fbx bone: {self.name}")

    def _set_transfer_animation(self):
        for sub_cls in _BipedBoneAnimationTransferBase.__subclasses__():
            ins = sub_cls(self)
            if ins.is_matched:
                self.transfer_animation = ins
                return
        else:
            raise NotImplementedError(f"Cannot find match to fbx bone: {self.name}")

    def _set_match_pose_to_fbx_bone(self):
        for sub_cls in _BipedBonePoseToFBXBoneMatcherBase.__subclasses__():
            ins = sub_cls(self)
            if ins.is_matched:
                self.match_pose_to_fbx_bone = ins
                return
        else:
            raise NotImplementedError(f"Cannot find match to fbx bone: {self.name}")

    def _set_match_guide(self):
        for sub_cls in _BipedBoneToFBXBoneMatchGuideBase.__subclasses__():
            ins = sub_cls(self)
            if ins.is_matched:
                self.match_guide = ins
                return
        else:
            raise NotImplementedError(f"Cannot find match to fbx bone: {self.name}")


class _FBXDummyBone(nodes._Dummy):
    """fbx를 import하면 joint가 dummy로 들어온다.
    
    dummy로 변환된 bone을 지칭한다.
    """
    _main_bone_names = ['root', 'pelvis', 'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'neck_01',
                        'neck_02', 'head',
                        'clavicle_l', 'clavicle_r',
                        'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r', 'hand_l', 'hand_r',
                        'thumb_01_l', 'thumb_01_r', 'thumb_02_l', 'thumb_02_r', 'thumb_03_l', 'thumb_03_r',
                        'index_metacarpal_l', 'index_metacarpal_r', 'index_01_l', 'index_01_r', 'index_02_l',
                        'index_02_r', 'index_03_l', 'index_03_r',
                        'middle_metacarpal_l', 'middle_metacarpal_r', 'middle_01_l', 'middle_01_r', 'middle_02_l',
                        'middle_02_r', 'middle_03_l', 'middle_03_r',
                        'ring_metacarpal_l', 'ring_metacarpal_r', 'ring_01_l', 'ring_01_r', 'ring_02_l', 'ring_02_r',
                        'ring_03_l', 'ring_03_r',
                        'pinky_metacarpal_l', 'pinky_metacarpal_r', 'pinky_01_l', 'pinky_01_r', 'pinky_02_l',
                        'pinky_02_r', 'pinky_03_l', 'pinky_03_r',
                        'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'ball_l', 'ball_r', 'toeEnd_l',
                        'toeEnd_r']

    def __init__(self, name: str, bones: '_FBXDummyBones'):
        super(_FBXDummyBone, self).__init__(name)
        self.bones = bones

    @property
    def is_main_bone(self):
        """rbf 같이 추가 되는 bone이 아닌 메인 bone인지 여부를 반환한다"""
        return self.name in self._main_bone_names

    def delete(self, with_children=False):
        super(_FBXDummyBone, self).delete(with_children)
        self.bones._bones.remove(self)


class _FBXMesh(nodes._Mesh):
    def __init__(self, meshes: '_FBXMeshes', name: str):
        super(_FBXMesh, self).__init__(name)
        self.meshes = meshes


class _FBXMeshes(Sequence[_FBXMesh]):
    def __init__(self, character: '_FBXCharacter'):
        self.char = character
        self._meshes = self._get_meshes_from_bones()

    def __repr__(self):
        return ', '.join(self.names)

    def __len__(self):
        return len(self._meshes)

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return self._meshes[index]
        else:
            raise NotImplementedError

    @property
    def names(self):
        return [mesh.name for mesh in self]

    @property
    def v(self):
        return [mesh.v for mesh in self]

    @v.setter
    def v(self, value):
        for mesh in self:
            mesh.v = value

    def hide(self):
        """모든 메쉬들을 끈다."""

    def _get_meshes_from_bones(self):
        meshes = []
        bone_names = self.char.bones.names
        for mesh in ncm.get_all_meshes():
            infs = ncm.get_influences(mesh)
            if len(set(infs) & set(bone_names)) > 0:
                meshes.append(_FBXMesh(self, mesh))
        return meshes


class _FBXDummyBones(Sequence[_FBXDummyBone]):
    """fbx를 하였을 때 dummy로 들어온 뼈들을 담는 클래스"""

    def __init__(self, character: '_FBXCharacter', root: str):
        self.char = character
        self._bones = self._get_bones_from_root(root)

    def __repr__(self):
        return ', '.join(self.names)

    def __len__(self):
        return self._bones

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return self._bones[index]
        else:
            raise NotImplementedError

    @property
    def names(self):
        return [bone.name for bone in self._bones]

    @property
    def v(self):
        return [bone.v for bone in self._bones]

    @v.setter
    def v(self, value):
        for mesh in self:
            mesh.v = value

    def _get_bones_from_root(self, root: str):
        bones: List[_FBXDummyBone] = []
        self._add_bone(root, bones)
        return bones

    def _add_bone(self, bone_name: str, bones):
        if ncm.get_class_of_object(bone_name) == rt.Dummy:
            bone = _FBXDummyBone(bone_name, self)
        elif ncm.get_class_of_object(bone_name) == rt.Point:
            return
        else:
            raise NotImplementedError(f"Type of {bone_name} is not supported")
        bones.append(bone)
        if bone.has_children:
            for child in bone.children:
                self._add_bone(child, bones)


class _BipedBones(Sequence[_BipedBone]):

    def __init__(self, biped: '_Biped'):
        self.biped = biped

    def __len__(self):
        return self._bones

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return self._bones[index]
        elif isinstance(index, str):
            for bone in self._bones:
                if bone.name == index:
                    return bone
                elif bone.pure_name == index:
                    return bone
            else:
                raise ValueError(f"Cannot find bone: {index}")
        else:
            raise NotImplementedError

    def __repr__(self):
        return ', '.join(self.names)

    @property
    def names(self):
        return [bone.name for bone in self._bones]

    @property
    def _bones(self):
        """biped의 모든 뼈를 반환한다
        
        com은 포함되지만 bone이 아닌 계산용 노드들은 제외된다.
        """
        bones = []
        bone = _BipedBone(self, self.biped.name)  # root
        bones.append(bone)

        if bone.has_children:
            self._add_children(bones, bone.name)
        return bones

    def find_bones_by_direction(self, direction: str = 'L'):
        """방향에 해당하는 bone들을 반환한다."""
        return [bone
                for bone in self._bones
                if bone.drt == direction]

    def find_bones_by_pure_name(self, name: str):
        """pure 이름에 해당 이름이 들어있는 bone들을 반환한다."""
        return [bone
                for bone in self._bones
                if name in bone.pure_name]

    def _add_children(self, bones, bone_name: str):
        bone_node = ncm.node(bone_name)
        for child in bone_node.children:
            child_node = ncm.get_node_by_name(child)
            if rt.ClassOf(child_node) == rt.Dummy:
                continue
            elif 'Footsteps' in child:
                continue
            else:
                bone = _BipedBone(self, child)
                bones.append(bone)
                if len(child_node.children) > 0:
                    self._add_children(bones, child_node.name)


class _BipedToFBXBoneLinker:
    def __init__(self, biped: '_Biped'):
        self.biped = biped

    def __call__(self):
        for bone in self.biped.bones:
            if bone.preset_fbx_bone_name is not None:
                ncm.link_constraint(bone.preset_fbx_bone_name, bone.name)

        self._link_spines()

    def _link_spines(self):
        """spine의 경우 preset에 등록이 안되어 있다.
        
        허리뼈의 갯수가 달라졌기 때문에 1:1 대응이 안된다.
        """
        fbx_spines = [ncm.dummy('spine_0' + str(i))
                      for i in range(2, 6)]
        spines = self.biped.bones.find_bones_by_pure_name('Spine')

        for i, spine in enumerate(spines):
            if i < len(spines) - 1:
                next_spine = spines[i + 1]
                target_fbx_spines = [fbx_spine
                                     for fbx_spine in fbx_spines
                                     if (fbx_spine.world_tz > spine.world_t[2]) and
                                     (fbx_spine.world_tz < next_spine.world_t[2])]

                for fbx_spine in fbx_spines:
                    if (fbx_spine.world_tz > spine.world_t[2]) and (fbx_spine.world_tz < next_spine.world_t[2]):
                        pass
                        # print('fbx_spine', fbx_spine)
                        # print('spine', spine) 
                        # print('next_spine', next_spine)
                        # print('fbx_spine.world_tz', fbx_spine.world_tz)
                        # print('spine.world_t[2]', spine.world_t[2])
                        # print('next_spine.world_t[2]', next_spine.world_t[2])
                        # print('--------------')


            else:
                target_fbx_spines = [fbx_spine
                                     for fbx_spine in fbx_spines
                                     if fbx_spine.world_tz > spine.world_t[2]]
            for target_fbx_spine in target_fbx_spines:
                ncm.link_constraint(target_fbx_spine.name, spine.name)


class _SpineTransformCalculater:
    """spine의 위치와 rotation을 계산한다
    
    지금은 3개 짜리 spine을 기준으로만 만들었다.
    """

    def __init__(self, biped: '_Biped'):
        self.biped = biped

    @property
    def size(self):
        return

    @property
    def ptns(self):
        return

    @property
    def rots(self):
        return


class _SpineTargetForAnimationTransfer:
    """spine의 위치를 계산할 target을 만든다"""

    def __init__(self, biped: '_Biped'):
        self.biped = biped

    @property
    def name(self):
        return 'spine_animation_target'

    def make(self):
        spine_target = ncm.point(self.name)
        clavicle_l = ncm.dummy('clavicle_l')
        clavicle_r = ncm.dummy('clavicle_r')
        last_spine = ncm.dummy('spine_05')

        spine_target.world_t = (clavicle_l.world_t + clavicle_r.world_t) / 2
        spine_target.world_r = last_spine.world_r
        spine_target.parent = last_spine

    def delete(self):
        ncm.point(self.name).delete()


class _Biped:
    """바이패드 클래스"""

    def __init__(self, name: str = None):
        """
        Attributes:
            self._preset: 바이패드 생성시 사용하는 preset
        """
        self._name = name
        self._make_preset = None
        self._bone_preset = None
        if ncm.exists_objects(name):
            self.node = ncm.get_node_by_name(name)
        else:
            self.node = None  # biped node
        self.bones = _BipedBones(self)
        self.link_to_fbx_character = _BipedToFBXBoneLinker(self)
        self.spine_target = _SpineTargetForAnimationTransfer(self)

    @property
    def name(self):
        return self._name

    @property
    def exists(self):
        """이름이 아직 안정해졌으면 없는거다."""
        if self.name is None:
            return False
        return ncm.exists_objects(self.name)

    @property
    def v(self):
        return [bone.v for bone in self.bones]

    @v.setter
    def v(self, value: bool):
        for bone in self.bones:
            bone.v = value

    @property
    def make_preset(self) -> _BipedMakePresetBase:
        return self._make_preset

    @make_preset.setter
    def make_preset(self, value: _BipedMakePresetBase):
        assert isinstance(value, _BipedMakePresetBase)
        self._make_preset = value

    @property
    def bone_preset(self) -> _BipedBonePresetBase:
        return self._bone_preset

    @bone_preset.setter
    def bone_preset(self, value: _BipedBonePresetBase):
        assert isinstance(value, _BipedBonePresetBase)
        self._bone_preset = value

    @property
    def figure_mode(self):
        """바이패드의 figure mode를 반환한다
        
        Notes:
            controller를 rt.classOf로 확인하면 
            Vertical_Horizontal_Turn 가 반환된다.
            이 controller에 figureMode등 biped의 주요 속성들이 들어있다.
        """
        return self.node.controller.figureMode

    @figure_mode.setter
    def figure_mode(self, value):
        self.node.controller.figureMode = value

    def delete(self):
        ncm.delete(self.name)

    def transfer_animation(self, start_frame=None, end_frame=None):
        """fbx 캐릭터의 애니메이션을 biped에 옮긴다"""
        if start_frame is None:
            start_frame = ncm.get_animation_range()[0]
        if end_frame is None:
            end_frame = ncm.get_animation_range()[1]

        self.spine_target.make()
        for bone in self.bones:
            bone.transfer_animation(start_frame, end_frame)
        # self.spine_target.delete()
        #         if bone.preset_fbx_bone_name is not None:
        #             ncm.link_constraint(bone.name, bone.preset_fbx_bone_name)
        # 
        # # with anim(True):
        # #     for f in range(0, 50):
        # #         with at(f):
        # #             for bone in self.bones:
        # #                 if bone.preset_fbx_bone_name is not None:
        # #                     fbx_bone = ncm.dummy(bone.preset_fbx_bone_name)
        # #                     bone.node.transform = fbx_bone.node.transform
        #         # fcount = fcount + 1
        #         # self.FitPoseByKeyFrame(self.mainBipedCom, frame=f)

    def make(self, preset: '_BipedMakePresetBase' = None, make_match_guide=False):
        """실제 씬에 biped를 생성한다.

        Args:
            preset: biped 생성시 사용할 preset
                make 옵션을 바꾸고 싶다면 preset을 새로 만들어서 집어넣어라
            make_match_guide: match할때 사용할 guide를 같이 만들어라.
        """
        if self.exists:
            print(f"Biped already exists: {self.name}")
            return
        if preset is not None:
            self.make_preset = preset

        self.node = rt.Biped.createNew(self.make_preset.height,
                                       self.make_preset.angle,
                                       rt.Point3(0, 0, self.make_preset.height),
                                       arms=self.make_preset.make_arms,
                                       spineLinks=self.make_preset.spine_joint_count,
                                       neckLinks=self.make_preset.neck_joint_count,
                                       fingers=self.make_preset.finger_count,
                                       fingerLinks=self.make_preset.finger_joint_count,
                                       knuckles=self.make_preset.metacarpal,
                                       shortThumb=self.make_preset.reduce_thumb_count,
                                       toeLinks=self.make_preset.toe_joint_count,
                                       triangleNeck=self.make_preset.triangle_neck)
        self._name = self.node.name

        if make_match_guide:
            self._make_match_guides()

    def match_to_fbx_character(self, scale=True, pose=True):
        """fbx 캐릭터에 맞춘다
        
        Notes:
            t, r을 맞추는 pose와 size를 맞추는 스케일은 각각 따로 작업을 해야 한다.
            setTransform으로 scale을 변경하면 다른 bone들의 스케일까지 영향을 준다.. 
            아주 욕나오게 이상한 구조를 가지고 있다.
        """
        self.spine_target.make()
        if scale:
            self._match_scale_to_fbx_character()
        if pose:
            self._match_pose_to_fbx_character()
        # self.spine_target.delete()

    def _match_scale_to_fbx_character(self):
        self.figure_mode = True
        for bone in self.bones:
            bone.match_scale_to_fbx_bone()
        self.figure_mode = False

    def _match_pose_to_fbx_character(self):
        for bone in self.bones:
            bone.match_pose_to_fbx_bone()

    def _make_match_guides(self):
        """biped의 위치를 수정할 때 사용할 match guide들도 같이 만든다.
        
        bone_preset에 따라 기본적인 위치를 잡아준다.
        """
        for bone in self.bones:
            if bone.is_available_match_guide:
                bone.match_guide.make()


class _BipedBoneToFBXBoneMatchGuideBase(ABC):

    def __init__(self, bone: _BipedBone):
        self.bone = bone

    @property
    def exists(self):
        """match guide를 만들었냐"""
        return ncm.exists_objects(self.name)

    @property
    @abstractmethod
    def is_available(self):
        """대상 bone에 match guide를 만들 수 있냐"""
        return

    @property
    @abstractmethod
    def is_matched(self):
        return

    @property
    def name(self):
        return f"MatchGuide_{self.bone.pure_name}"

    def make(self):
        """match guide를 만든다"""
        if not self.is_available:
            print(f"Match guide is not available: {self.name}")
            return
        self._make_dummy()
        self._set_position()

    @abstractmethod
    def _set_position(self):
        pass

    def _make_dummy(self):
        dummy = rt.Dummy()
        dummy.name = self.name

    def _get_spine_gab(self):
        clavicle_tz = ncm.get_node_by_name('clavicle_l').transform.position.z
        spine_1_tz = ncm.get_node_by_name('spine_01').transform.position.z
        return (clavicle_tz - spine_1_tz) / 3


class _Spine1BipedBoneToFBXBoneMatchGuide(_BipedBoneToFBXBoneMatchGuideBase):

    @property
    def is_available(self):
        return ncm.exists_objects('spine_01')

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Spine1'

    def _set_position(self):
        spine_1_tz = ncm.get_node_by_name('spine_01').transform.position.z
        ptn = rt.Point3(0, 0, spine_1_tz + self._get_spine_gab())
        ncm.get_node_by_name(self.name).position = ptn


class _Spine2BipedBoneToFBXBoneMatchGuide(_BipedBoneToFBXBoneMatchGuideBase):

    @property
    def is_available(self):
        return ncm.exists_objects('spine_01')

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Spine2'

    def _set_position(self):
        spine_1_tz = ncm.get_node_by_name('spine_01').transform.position.z
        ptn = rt.Point3(0, 0, spine_1_tz + self._get_spine_gab() * 2)
        ncm.get_node_by_name(self.name).position = ptn


class _UnavailableBipedBoneToFBXBoneMatchGuide(_BipedBoneToFBXBoneMatchGuideBase):
    """match guide가 필요하지 않는 bone들에 사용하는 공통 클래스"""

    @property
    def is_matched(self):
        return True

    @property
    def is_available(self):
        return False

    def _set_position(self):
        pass


class _FBXCharacter:
    """fbx로 가져온 character를 지칭한다
    
    fbx로 가져온 파일을 정리하거나 converting하는데 쓴다.
    """

    def __init__(self, root: str):
        if not ncm.exists_objects(root):
            print(f"Object does not exist: {root}")
            self._exists = False
            return
        self.bones = _FBXDummyBones(self, root)
        self.meshes = _FBXMeshes(self)
        self._exists = True

    def __repr__(self):
        """특별한 이름은 없으므로 bone들을 반환한다.
        
        나중에는 캐릭터 여러마리를 fbx로 씬에 넣어야 할 수도 있으니
        namespace를 넣을 수 있게 되면 그걸로 바꿔야 할지도
        """
        return str(self.bones)

    @property
    def exists(self):
        return self._exists

    def delete_sub_bones(self):
        """names에 해당하는 뼈를 삭제한다"""
        for bone in reversed(self.bones._bones):
            if not bone.is_main_bone:
                bone.delete()
