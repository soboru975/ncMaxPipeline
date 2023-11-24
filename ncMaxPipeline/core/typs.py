from dataclasses import dataclass
from pymxs import runtime as rt


@dataclass
class Typ:
    DUMMY = rt.Dummy
    BONE = rt.Bone