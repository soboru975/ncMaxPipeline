from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Union, List, Optional

import numpy as np
from pymxs import runtime as rt
import ncMaxPipeline as ncm
from ncMaxPipeline.rig import nodes


def biped(name: str = 'Bip001'):
    bp = _Biped(name)
    bp.make_preset = _BipedMakePresets.PROJECT_M
    bp.bone_preset = _BipedBonePresets.PROJECT_M
    return bp


def get_fbx_character(root: str = 'root'):
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
    LowerArm = _BipedBoneData('lowerarm', True)
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


class _BipedBoneToFBXBoneMatcherBase(ABC):
    def __init__(self, bone: '_BipedBone'):
        self.bone = bone

    def __call__(self):
        if not self.biped.figure_mode:
            print(f"Figure mode is not on. Matchings will not be executed: {self.bone.name}")
            return
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


class _ComToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

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
        """
        thigh_l = ncm.get_object_by_name('thigh_l')
        thigh_r = ncm.get_object_by_name('thigh_r')

        avg_ptn = (thigh_l.transform.position + thigh_r.transform.position) / 2
        self.bone.world_t = 0, avg_ptn[1], avg_ptn[2]
        # ptn = rt.Point3(0, avg_ptn[1], avg_ptn[2])
        # rt.biped.setTransform(self.bone.node, rt.Name('pos'), ptn, False)


class _HeadBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Head'

    def _execute_match(self):
        pass


# class _ToeBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):
# 
#     @property
#     def is_matched(self):
#         return self.bone.pure_name == 'Toe'
# 
#     def _execute_match(self):
#         pass


class _SpineBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Spine'

    def _execute_match(self):
        pass


class _FingerBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return 'Finger' in self.bone.pure_name

    def _execute_match(self):
        pass


# 
# class _FootBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):
# 
#     @property
#     def is_matched(self):
#         return 'Foot' == self.bone.pure_name
# 
#     def _execute_match(self):
#         fbx_bone_name = self.bone.pure_name.lower()
#         if self.bone.drt == 'L':
#             fbx_bone = ncm.dummy(fbx_bone_name + '_l')
#         else:
#             fbx_bone = ncm.dummy(fbx_bone_name + '_r')
#         self.bone.world_t = fbx_bone.world_t


class _PelvisBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return 'Pelvis' in self.bone.pure_name

    def _execute_match(self):
        """pelvis 크기 설정

        Notes:
            여기서 사이즈 조절을 해주는 것은 오로지 thigh의 위치를 맞추기 위한 
            이유 이다. pelvis를 가로로 늘려서 biped의 thigh의 위치를 맞추는 것이다.
        """
        thigh_l = ncm.get_object_by_name('thigh_l')
        thigh_r = ncm.get_object_by_name('thigh_r')

        thigh_gap = thigh_l.transform.position.x - thigh_r.transform.position.x
        scale = rt.Point3(thigh_gap, thigh_gap, thigh_gap)
        rt.biped.setTransform(self.bone.node, rt.Name('scale'), scale, False)


class _NeckBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return 'Neck' in self.bone.pure_name

    def _execute_match(self):
        pass


class _ClavicleBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):

    @property
    def is_matched(self):
        return 'Clavicle' in self.bone.pure_name

    def _execute_match(self):
        pass


class _ThighBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):
    """일반적인 뼈들을 fbx뼈로 매칭시킨다

    _BipedBoneToFBXBoneMatcherBase.__subclasses__() 반복문을 돌때
    설정되지 못한 bone들을 매칭시킨다.
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
        self.bone.world_r = fbx_bone.world_r
        if self.bone.drt == 'L':
            self.bone.rz = 180

    def _match_size(self, fbx_bone: nodes._Dummy):
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child_fbx_bone.world_t)
        self.bone.size = bone_length


class _CalfBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):
    """일반적인 뼈들을 fbx뼈로 매칭시킨다
    
    _BipedBoneToFBXBoneMatcherBase.__subclasses__() 반복문을 돌때
    설정되지 못한 bone들을 매칭시킨다.
    """

    @property
    def is_matched(self):
        return self.bone.pure_name in ['Calf']

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
        self._rotate_calf(fbx_bone)
        # self._adjust_rotation(fbx_bone)

    # def _adjust_rotation(self, fbx_bone: nodes._Dummy):
    #     """로테이션을 보정한다.
    #     
    #     이유는 모르겠는데 에러가 나서 그걸 보정한다.
    #         피규어 모드가 켜져있는 상태에서 스크립트로 회전을 주고 난 다음에 피규어 모드를 끄게 될것이다.
    #         그러면 parent coordinate system을 기준으로 약간의 회전값들이 들어가 있다.
    #         피규어 모드를 끈 상태에서 parent coordinate system상의 rotation을 모두 0, 0, 0으로 만들면
    #         원하는 모양이 되기는 한다. 그래서 그 정도의 값을 보정해 준다. 되려나  
    #     """
    #     parent_node = self.bone.node.parent
    #     rot = (self.bone.node.transform * rt.Inverse(parent_node.transform)).rotation
    #     rz = rt.QuatToEuler(rot).z
    #     self.bone.rz -= rz

    def _rotate_calf(self, fbx_bone: nodes._Dummy):
        """calf의 rotation을 조절한다
        
        원래 계획은
            thigh의 회전과 크기를 맞춘다 -> calf의 크기를 맞춘다 -> foot의 위치를 맞춘다.
            
        이었는데 foot의 위치를 바꾸는 순간 무릎의 위치가 엇나가게 된다.
        그러므로 다음과 같이 순서를 수정하였다.
            thigh의 회전과 크기를 맞춘다 -> calf의 축 하나만 돌려서 foot의 위치를 맞춘다
            
        이 방법으로 하려면 vector로 각도를 구해야 한다.
        """
        if self.bone.drt == 'L':
            biped_foot = self.biped.bones[f'{self.biped.name} L Foot']
            fbx_foot = ncm.dummy('foot_l')
        else:
            biped_foot = self.biped.bones[f'{self.biped.name} R Foot']
            fbx_foot = ncm.dummy('foot_r')
        vec_1 = fbx_bone.world_t - fbx_foot.world_t
        vec_2 = fbx_bone.world_t - biped_foot.world_t
        angle = ncm.get_angle_between_vectors(vec_1, vec_2)
        self.bone.rz += angle

    def _match_size(self, fbx_bone: nodes._Dummy):
        child_fbx_bone = ncm.dummy(fbx_bone.children[0])
        bone_length = np.linalg.norm(fbx_bone.world_t - child_fbx_bone.world_t)
        self.bone.size = bone_length


class _NotSetYetBoneToFBXBoneMatcher(_BipedBoneToFBXBoneMatcherBase):
    """아직 설정하지 않은 bone들
    """

    @property
    def is_matched(self):
        return True

    def _execute_match(self):
        print(f"Not set yet: {self.bone.name}")


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
        self._set_match_to_fbx_bone()

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
        return self.node.rotation

    @r.setter
    def r(self, value):
        world_rot_mat = rt.Matrix3(1)
        world_rot_mat.rotation = self.world_r
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
        else:
            raise ValueError(type(value))

    @property
    def world_r(self):
        return self.node.transform.rotation

    @world_r.setter
    def world_r(self, value):
        """biiped의 bone의 rotation을 설정하는 방법은 약간 다르다"""
        if isinstance(value, (np.ndarray, tuple, list)):
            rot = rt.EulerToQuat(rt.EulerAngles(float(value[0]), float(value[1]), float(value[2])))
            rt.biped.setTransform(self.node, rt.Name('rotation'), rot, False)
        elif isinstance(value, rt.Rotation):
            rt.biped.setTransform(self.node, rt.Name('rotation'), value, False)
        else:
            raise ValueError(type(value))

    @property
    def size(self):
        return

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
        return splits[1]

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

    def _set_match_to_fbx_bone(self):
        for sub_cls in _BipedBoneToFBXBoneMatcherBase.__subclasses__():
            ins = sub_cls(self)
            if ins.is_matched:
                self.match_to_fbx_bone = ins
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
        if ncm.get_type_of_object(bone_name) == ncm.Typ.DUMMY:
            bone = _FBXDummyBone(bone_name, self)
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

    def _add_children(self, bones, bone_name: str):
        bone_node = ncm.node(bone_name)
        for child in bone_node.children:
            child_node = ncm.get_object_by_name(child)
            if rt.ClassOf(child_node) == rt.Dummy:
                continue
            elif 'Footsteps' in child:
                continue
            else:
                bone = _BipedBone(self, child)
                bones.append(bone)
                if len(child_node.children) > 0:
                    self._add_children(bones, child_node.name)


class _BipedBoneToFBXBoneMatcher:
    def __init__(self, biped: '_Biped'):
        self.biped = biped
        self.match_com_to_fbx_bone = _ComToFBXBoneMatcher()
        self.match_com_to_fbx_bone = _ComToFBXBoneMatcher()

    def __call__(self, char: '_FBXCharacter'):
        if not char.exists:
            print(f"Character does not exist: {char}")
            return
        self.char = char
        self.figure_mode = True
        self.match_com_to_fbx_bone()

        self.figure_mode = False


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
            self.node = ncm.get_object_by_name(name)
        else:
            self.node = None  # biped node
        self.bones = _BipedBones(self)
        self.match_to_fbx_character = _BipedBoneToFBXBoneMatcher(self)

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

    def make(self, preset: '_BipedMakePresetBase' = None, make_match_guide=True):
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
                                       toeLinks=self.make_preset.toe_joint_count)
        self._name = self.node.name

        if make_match_guide:
            self._make_match_guides()

    def match_to_fbx_character(self, char: '_FBXCharacter'):
        if not char.exists:
            print(f"Character does not exist: {char}")
            return
        self.char = char
        self.figure_mode = True
        for bone in self.bones:
            bone.match_to_fbx_bone()
            if bone.name == 'Bip001 R Calf':
                break
        # self.figure_mode = False

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
        clavicle_tz = ncm.get_object_by_name('clavicle_l').transform.position.z
        spine_1_tz = ncm.get_object_by_name('spine_01').transform.position.z
        return (clavicle_tz - spine_1_tz) / 3


class _Spine1BipedBoneToFBXBoneMatchGuide(_BipedBoneToFBXBoneMatchGuideBase):

    @property
    def is_available(self):
        return ncm.exists_objects('spine_01')

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Spine1'

    def _set_position(self):
        spine_1_tz = ncm.get_object_by_name('spine_01').transform.position.z
        ptn = rt.Point3(0, 0, spine_1_tz + self._get_spine_gab())
        ncm.get_object_by_name(self.name).position = ptn


class _Spine2BipedBoneToFBXBoneMatchGuide(_BipedBoneToFBXBoneMatchGuideBase):

    @property
    def is_available(self):
        return ncm.exists_objects('spine_01')

    @property
    def is_matched(self):
        return self.bone.pure_name == 'Spine2'

    def _set_position(self):
        spine_1_tz = ncm.get_object_by_name('spine_01').transform.position.z
        ptn = rt.Point3(0, 0, spine_1_tz + self._get_spine_gab() * 2)
        ncm.get_object_by_name(self.name).position = ptn


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
