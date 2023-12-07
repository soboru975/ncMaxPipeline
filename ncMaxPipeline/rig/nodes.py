from abc import abstractmethod, ABC
from typing import List, Optional

import ncMaxPipeline as ncm
import numpy as np
from pymxs import runtime as rt


def node(name: str):
    return _Node(name)


def dummy(name: str):
    return _Dummy(name)


class _Object(ABC):
    """
    그냥 node에 포함하면 되지 왜 이걸 굳이 나눴냐 궁금할 수 있다.
    tx, ty.. 같은 property들이 일반 node와 biped bone이 적용방식이 서로 너무 달랐다.
    그래서 node의 tx, ty.. 같은 property들이 상속되기를 원치 않아서 나누었다.
    """

    def __init__(self, name: str):
        self.node = ncm.get_node_by_name(name)
        if not self.exists:
            self.make(name)

    def __repr__(self):
        return self.name

    @property
    def name(self):
        return self.node.name

    @property
    def v(self):
        """visibility"""
        return not self.node.isHidden

    @v.setter
    def v(self, value):
        self.node.isHidden = not value

    @property
    def parent(self) -> Optional[str]:
        if self.node.parent is None:
            return None
        else:
            return self.node.parent.name

    @parent.setter
    def parent(self, value):
        if isinstance(value, _Object):
            self.node.parent = value.node
        elif isinstance(value, str):
            self.node.parent = ncm.get_node_by_name(value)
        else:
            raise ValueError(value, type(value))

    @property
    def has_parent(self):
        return self.node.parent is not None

    @property
    def has_children(self):
        return len(self.node.children) > 0

    @property
    def children(self) -> List[str]:
        return [child.name for child in self.node.children]

    @property
    def grand_children(self) -> List[str]:
        """모든 하위 계층의 자식들을 반환한다."""
        children = []
        for child in self.children:
            self._add_children(child, children)
        return children

    @property
    def exists(self):
        if self.node is None:
            return False
        if self.name is None:
            return False
        if ncm.exists_objects(self.name):
            return True
        else:
            return False


class _Node(_Object):
    """(현재 개념이 정확하지는 않지만) max는 노드 기반이 아니기 때문에
    노드라는 개념은 씬에 존재하는 object만을 뜻하는 것으로 한다.
    
    deform 스택 등은 노드가 아니다.
    """

    @property
    def t(self):
        if self.node.parent is None:
            ptn = self.node.position
            return np.array([ptn.x, ptn.y, ptn.z])
        else:
            inv_parent_transform = rt.Inverse(self.node.parent.transform)
            ptn = self.node.position * inv_parent_transform
            return np.array([ptn.x, ptn.y, ptn.z])

    @t.setter
    def t(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            ptn = rt.Point3(float(value[0]), float(value[1]), float(value[2]))
            if self.node.parent is None:
                self.node.position = ptn
            else:
                self.node.position = ptn * self.node.parent.transform
        else:
            raise ValueError(type(value))

    @property
    def tx(self):
        return float(self.t[0])

    @tx.setter
    def tx(self, value):
        t = self.t
        t[0] = value
        self.t = t

    @property
    def ty(self):
        return float(self.t[1])

    @ty.setter
    def ty(self, value):
        t = self.t
        t[1] = value
        self.t = t

    @property
    def tz(self):
        return float(self.t[2])

    @tz.setter
    def tz(self, value):
        t = self.t
        t[2] = value
        self.t = t

    @property
    def r(self):
        if self.node.parent is None:
            rot = rt.QuatToEuler(self.node.transform.rotation)
            return np.array([rot.x, rot.y, rot.z])
        else:
            inv_parent_transform = rt.Inverse(self.node.parent.transform)
            mat = rt.Matrix3(1)
            mat.rotation = self.node.transform.rotation
            rot = rt.QuatToEuler((mat * inv_parent_transform).rotation)
            return np.array([rot.x, rot.y, rot.z])

    @r.setter
    def r(self, value):
        """
        
        Notes:
            사실 아직도 잘 모르겠다. 
            rotation을 주면 위치값도 변해 버려서 위치값을 다시 처음 값으로 지정해 줘야 한다.
            왜 로컬 을 주는데 inverse를 하는지 난 잘 모르겠다. 어쨌든 inverse를 하니 제대로된 값이 들어가기는 하는것 같다.
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            rot = rt.EulerToQuat(rt.EulerAngles(float(value[0]), float(value[1]), float(value[2])))
            if self.node.parent is None:
                transform = self.node.transform
                mat = rt.Matrix3(1)
                mat.rotation = rot
                self.node.rotation = rt.Inverse(mat).rotation
                self.node.position = transform.position
            else:
                t = self.t
                parent_transform = self.node.parent.transform
                local_mat = rt.Matrix3(1)
                local_mat.rotation = rot
                self.node.rotation = rt.Inverse((local_mat * parent_transform)).rotation
                self.t = t
        else:
            raise ValueError(type(value))

    @property
    def rx(self):
        return self.r[0]

    @rx.setter
    def rx(self, value):
        self.r = [value, self.ry, self.rz]

    @property
    def ry(self):
        return self.r[1]

    @ry.setter
    def ry(self, value):
        self.r = [self.rx, value, self.rz]

    @property
    def rz(self):
        return self.r[2]

    @rz.setter
    def rz(self, value):
        self.r = [self.rx, self.ry, value]

    @property
    def s(self):
        if self.node.parent is None:
            return np.array([self.node.scale.x, self.node.scale.y, self.node.scale.z])
        else:
            local_mat = rt.Matrix3(1)
            local_mat.scale = self.node.scale
            scale = (local_mat * self.node.parent.transform).scale
            return np.array([scale.x, scale.y, scale.z])

    @s.setter
    def s(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            scale = rt.Point3(float(value[0]), float(value[1]), float(value[2]))
            if self.node.parent is None:
                self.node.scale = scale
            else:
                parent_transform = self.node.parent.transform
                local_mat = rt.Matrix3(1)
                local_mat.scale = scale
                self.node.scale = (local_mat * parent_transform).scale
        else:
            raise ValueError(type(value))

    @property
    def sx(self):
        return self.node.scale.x

    @sx.setter
    def sx(self, value):
        self.node.scale.x = value

    @property
    def sy(self):
        return self.node.scale.y

    @sy.setter
    def sy(self, value):
        self.node.scale.y = value

    @property
    def sz(self):
        return self.node.scale.z

    @sz.setter
    def sz(self, value):
        self.node.scale.z = value

    @property
    def world_t(self) -> np.ndarray:
        return np.array([self.world_tx, self.world_ty, self.world_tz])

    @world_t.setter
    def world_t(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            self.node.position = rt.Point3(float(value[0]), float(value[1]), float(value[2]))
        elif rt.ClassOf(value) == rt.Point3:
            self.node.position = value
        else:
            raise ValueError(type(value))

    @property
    def world_tx(self):
        return self.node.transform.position.x

    @world_tx.setter
    def world_tx(self, value):
        ptn = rt.Point3(float(value), self.world_ty, self.world_tz)
        self.node.position = ptn

    @property
    def world_ty(self):
        return self.node.position.y

    @world_ty.setter
    def world_ty(self, value):
        ptn = rt.Point3(self.world_tx, float(value), self.world_tz)
        self.node.position = ptn

    @property
    def world_tz(self):
        return self.node.position.z

    @world_tz.setter
    def world_tz(self, value):
        ptn = rt.Point3(self.world_tx, self.world_ty, float(value))
        self.node.position = ptn

    @property
    def world_r(self):
        rot = rt.QuatToEuler(self.node.transform.rotation)
        return np.array([rot.x, rot.y, rot.z])

    @world_r.setter
    def world_r(self, value):
        """world rotation 적용.
        
        이게 원래 self.node.rotation으로 값을 넣었는데 값이 삐뚤어져서 들어간다.
        그래서 이유는 모르겠지만 matrix로 변환해서 값을 넣었다.
        
        matrix를 만들때도 matrix에 .position을 먼저 입력하면 안된다.
        아래처럼 rotation을 먼저 입력해야 한다.
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            mat = rt.Matrix3(1)
            mat.rotation = rt.EulerToQuat(rt.EulerAngles(float(value[0]), float(value[1]), float(value[2])))
            mat.position = rt.Point3(self.world_tx, self.world_ty, self.world_tz)
            self.node.transform = mat
        else:
            raise ValueError(type(value))

    @property
    def world_rx(self):
        return rt.QuatToEuler(self.node.rotation).x

    @world_rx.setter
    def world_rx(self, value):
        """아직 검증안됨 구문"""
        self.node.transform.rotation.x = value

    @property
    def world_ry(self):
        return rt.QuatToEuler(self.node.rotation).y

    @world_ry.setter
    def world_ry(self, value):
        """아직 검증안됨 구문"""
        self.node.transform.rotation.y = value

    @property
    def world_rz(self):
        return rt.QuatToEuler(self.node.rotation).z

    @world_rz.setter
    def world_rz(self, value):
        """아직 검증안됨 구문"""
        self.node.transform.rotation.z = value

    @property
    def world_s(self):
        return np.array([self.world_sx, self.world_sy, self.world_sz])

    @world_s.setter
    def world_s(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            self.world_sx = value[0]
            self.world_sy = value[1]
            self.world_sz = value[2]
        else:
            raise ValueError(type(value))

    @property
    def world_sx(self):
        return self.node.transform.scale.x

    @world_sx.setter
    def world_sx(self, value):
        self.node.transform.scale.x = value

    @property
    def world_sy(self):
        return self.node.transform.scale.y

    @world_sy.setter
    def world_sy(self, value):
        self.node.transform.scale.y = value

    @property
    def world_sz(self):
        return self.node.transform.scale.z

    @world_sz.setter
    def world_sz(self, value):
        self.node.transform.scale.z = value

    @property
    def mat(self):
        if self.node.parent is None:
            return self.node.transform
        else:
            return self.node.transform * rt.Inverse(self.node.parent.transform)

    @mat.setter
    def mat(self, value):
        if self.node.parent is None:
            self.node.transform = value
        else:
            self.node.transform = value * self.node.parent.transform

    @property
    def world_mat(self):
        return self.node.transform

    @world_mat.setter
    def world_mat(self, value):
        self.node.transform = value

    def make(self, name: str):
        pass

    def make_grp(self, group_name: str = 'grp') -> '_Dummy':
        """그룹을 만들어준다."""
        grp = ncm.dummy(self.name + '_' + group_name)
        grp.world_t = self.world_t
        grp.world_r = self.world_r
        
        if self.parent is not None:
            grp.parent = self.parent
        self.parent = grp.name 
        return grp

    def delete(self, with_children=False):
        """노드를 지워준다.
        
        지울때 마야와 약간 다르다
        max는 기본적으로 delete()해주면 하위의 오브젝트들을 같이 지우는게 아니고
        world 하이라키로 반환해준다. 본인만 지워진다.
        """
        if with_children:
            for child in self.grand_children:
                ncm.delete(child.name)
        if self.exists:
            ncm.delete(self.name)

    def _add_children(self, child, children: list):
        children.append(child)
        child_node = ncm.node(child)
        for child in child_node.children:
            self._add_children(child, children)


class _Bone(_Node):
    pass


class _Dummy(_Node):
    def make(self, name: str):
        self.node = rt.Dummy()
        self.node.name = name


class _Mesh(_Node):
    pass
