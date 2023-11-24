import os
import shutil
import sys
from math import pi
from typing import List

import numpy as np
from pymxs import runtime as rt
import ncMaxPipeline as ncm


def exists_file(path: str):
    return rt.DoesFileExist(path)


def select(names):
    if names is None:
        return
    assert isinstance(names, str) or isinstance(names, list), \
        f"Names must be string or list: {names}, {type(names)}"
    if isinstance(names, str):
        names = [names]
    for name in names:
        if exists_objects(name):
            rt.select(get_object_by_name(name))


def open_file(path: str = None):
    """
    rt.Name("noPrompt")라고 넣는 것은 새 씬을 만들때 
    저장 여부를 사용자에게 물어보지 않는다.
    """
    if path is None:
        rt.resetMaxFile(rt.Name("noPrompt"))
        rt.Execute("viewport.activeViewport = 4")
        rt.Execute("max tool maximize")
        return
    if not exists_file(path):
        print(f"File does not exist: {path}")
        return
    rt.LoadMaxFile(path)


def get_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray):
    assert isinstance(vec1, np.ndarray), f"Vec1 must be numpy array: {vec1}, {type(vec1)}"
    assert isinstance(vec2, np.ndarray), f"Vec2 must be numpy array: {vec2}, {type(vec2)}"
    assert vec1.shape == vec2.shape, f"Shape must be same: {vec1.shape}, {vec2.shape}"

    length_vec1 = np.linalg.norm(vec1)
    length_vec2 = np.linalg.norm(vec2)
    vec1 = vec1 / length_vec1
    vec2 = vec2 / length_vec2
    dot_product = np.dot(vec1, vec2)

    return np.arccos(dot_product) * 180.0 / pi


def exists_objects(names):
    if names is None:
        return False
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name is None:
            return False
        node = rt.getNodeByName(name)
        if node is None:
            return False
    return True


def toggle_meshes_see_through(meshes=None):
    """모든 mesh들을 see through상태로 만든다."""
    if meshes is None:
        objs = rt.objects
    else:
        objs = [rt.getNodeByName(mesh) for mesh in meshes]

    init = None

    for obj in objs:
        if rt.classOf(obj) in [rt.Editable_Mesh, rt.Editable_Poly, rt.PolyMeshObject]:
            if init is None:
                init = rt.getProperty(obj, "xray")
            if init:
                rt.setProperty(obj, "xray", False)
            else:
                rt.SetProperty(obj, "xray", True)


def get_all_meshes() -> List[str]:
    return [obj.name
            for obj in rt.objects
            if rt.isKindOf(obj, rt.GeometryClass)]


def delete(names):
    if isinstance(names, str):
        names = [names]
    for name in names:
        if exists_objects(name):
            rt.delete(get_object_by_name(name))


def get_type_of_object(name: str):
    """노드의 type을 받는다.
    
    Examples:
        typ = ncm.get_type_of_object('Bip001 R Finger0Nub')
        print(typ == ncm.Typ.DUMMY)
    """
    return rt.classOf(get_object_by_name(name))


def save_incremental():
    current_filepath = rt.maxFilePath + rt.maxFileName

    if current_filepath == "":
        rt.messageBox("File is not saved yet. Please save the file first.",
                      title="Incremental Save Error", beep=True)
        return

    # 파일 경로와 파일명 분리
    dir_path, filename = os.path.split(current_filepath)
    basename, extension = os.path.splitext(filename)

    # 'incrementalSave' 폴더 생성
    inc_save_dir = os.path.join(dir_path, 'incrementalSave')
    if not os.path.exists(inc_save_dir):
        os.makedirs(inc_save_dir)

    # 파일명으로 된 폴더 생성
    file_specific_dir = os.path.join(inc_save_dir, basename)
    if not os.path.exists(file_specific_dir):
        os.makedirs(file_specific_dir)

    # 현재 파일을 그대로 저장
    rt.saveMaxFile(current_filepath)

    # 파일 리스트 가져오기 및 가장 큰 버전 번호 찾기
    files = os.listdir(file_specific_dir)
    version = 1
    if files:
        highest_version = max([int(re.search(r"_(\d+)", f).group(1)) for f in files if re.search(r"_(\d+)", f)])
        version = highest_version + 1

    new_filepath = os.path.join(file_specific_dir, f"{basename}_{version:03}{extension}")
    shutil.copy2(current_filepath, new_filepath)
    return new_filepath


def print_previous_functions(num=5):
    def get_previous_function_name(num=2):
        full_len = 30
        cls_name = ''

        try:
            if 'self' in sys._getframe(num).f_locals:
                cls_name = sys._getframe(num).f_locals['self'].__class__.__name__

            func_name = sys._getframe(num).f_code.co_name
            module_name = (sys._getframe(num).f_globals['__name__']).split('.')[-1]
            if cls_name:
                module_name += '.' + cls_name

            string = func_name + ''.join([' '] * (full_len - len(func_name)))
            string = string + '--    ' + module_name
            return string

        except:
            return "error"

    print('-----------------------------------')
    for i in range(num):
        history = num + 2 - i
        print(history - 2, get_previous_function_name(num=history))


def get_object_by_name(name: str):
    assert isinstance(name, str), f"Name must be string: {name}, {type(name)}"
    return rt.getNodeByName(name)


def get_bound_meshes_by_dummy(dummy: str) -> List[str]:
    """더미 이름을 넣으면 더미가 바인딩된 메쉬를 찾는다"""
    meshes = []
    for node in rt.objects:
        for mod in node.modifiers:
            if rt.classOf(mod) == rt.Skin:
                for bone in rt.skinOps.GetBoneNodes(mod):
                    if bone.name == dummy:
                        meshes.append(node.name)
                        break
    return meshes


def is_skinned_mesh(mesh: str) -> bool:
    """메쉬가 스킨되어 있는지 확인한다."""
    for mod in get_object_by_name(mesh).modifiers:
        if rt.classOf(mod) == rt.Skin:
            return True
    else:
        return False


def get_skin_node(mesh: str):
    for mod in get_object_by_name(mesh).modifiers:
        if rt.classOf(mod) == rt.Skin:
            return mod
    else:
        return None


def get_influences(mesh: str) -> List[str]:
    """mesh를 넣으면 influence들을 반환한다.
    
    max에서의 용어는 inf가 아닌거 같지만 일단 이렇게 하자.
    나중에 맞는 용어를 찾으면 수정할 것
    """
    if not is_skinned_mesh(mesh):
        print(f"Mesh has no skin: {mesh}")
        return []
    skin = get_skin_node(mesh)
    return [bone.name for bone in rt.skinOps.GetBoneNodes(skin)]


def unload_packages(print_result=False):
    packages = ['ncMaxPipeline']
    reload_list = []
    for i in sys.modules.keys():
        for package in packages:
            if i.startswith(package):
                reload_list.append(i)

    for i in reload_list:
        try:
            if sys.modules[i] is not None:
                del (sys.modules[i])
                if print_result:
                    print(f"Unloaded: {i}")
        except:
            print(f"Failed to unload: {i}")
