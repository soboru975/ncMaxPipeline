from dataclasses import dataclass
from typing import Optional

import ncMaxPipeline as ncm
import numpy as np
from pymxs import runtime as rt

@dataclass
class _SinglePreset:
    target: str
    rotate: bool
    
@dataclass
class _AveragePreset:
    targets: list(str)
    rotate: bool

@dataclass
class _PointData:
    point: Optional[str]
    side: str = "L" #options: "L", "R", "Mid"

@dataclass
class _PresetBase:
    Center = _SinglePreset("Bip002 Spine1", False)
    Shoulder = _AveragePreset(["Bip002 L Clavicle", "Bip002 L UpperArm"], True)

    @property
    def point_name(self):
        names = []
        for attr in dir(self):
            names = []
            if isinstance(getattr(type(self), attr), (_SinglePreset, _AveragePreset)):
                '''
                getattr()의 값들 중에서
                내가 _PresetBase 클래스에서 추가한 변수만 취하기 위해
                클래스 타입을 체크한 것임
                '''
                names += f'Point {attr}'
            return names


class Main:
    def __init__(self, name: str):
        self.node = ncm.get_node_by_name(name)
        
    def _make_point(self):
        pass
        
    @property
    def LinkConstraint(self):
        pass

    @LinkConstraint.setter
    def LinkConstraint(self):
        pass




