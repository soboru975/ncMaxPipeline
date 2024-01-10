# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from enum import Enum
import ncMaxPipeline as ncm

from pymxs import runtime as rt


def install_project_m_menu():
    _Menu()


class _Category:
    BIPED = 'Biped'


class _SubMenuBase(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        pass

    @property
    def tooltip(self):
        return ''

    @property
    @abstractmethod
    def button(self):
        pass

    @property
    def command_path(self):
        """자동으로 command path를 반환한다."""
        user_scripts_path = rt.GetDir(rt.Name('userScripts'))
        commands_path = user_scripts_path + '\\ncMaxPipeline\\UI\\projectM_menu\\commands\\'
        path = commands_path + self.name + '.py'
        return path.replace('\\', '\\\\')
    
    @property
    def command(self):
        return ''


class _CreateBiped(_SubMenuBase):

    @property
    def name(self):
        return 'CreateBiped'

    @property
    def button(self):
        return 'Create Biped'

    @property
    def category(self):
        return _Category.BIPED


class _MatchToFBXBones(_SubMenuBase):

    @property
    def name(self):
        return 'MatchToFBXBones'

    @property
    def category(self) -> str:
        return _Category.BIPED

    @property
    def button(self):
        return 'Match Biped to FBX Bones'


class _SubMenus:
    def __init__(self, menu: '_Menu'):
        self.menu = menu

    @property
    def main_menu(self):
        return self.menu.main_menu

    def add(self):
        for sub_menu_cls in _SubMenuBase.__subclasses__():
            sub_menu = sub_menu_cls()
            macroscript_content = f'python.ExecuteFile ("{sub_menu.command_path}")'

            rt.macros.new(sub_menu.category,
                          sub_menu.name,
                          sub_menu.tooltip,
                          sub_menu.button,
                          macroscript_content)
            sub_menu = rt.menuMan.createActionItem(sub_menu.name, sub_menu.category)
            self.main_menu.addItem(sub_menu, -1)



class _Menu:

    def __init__(self):
        self._set_variables()
        self._delete_existed_menu()
        self._add_main_menu()
        self._add_sub_menus()
        rt.menuMan.updateMenuBar()

    def _set_variables(self):
        self.main_menu_name = 'Project M'
        self.sub_menus = _SubMenus(self)

    def _delete_existed_menu(self):
        old_menu = rt.menuMan.findMenu(self.main_menu_name)
        if old_menu:
            rt.menuMan.unRegisterMenu(old_menu)

    def _add_sub_menus(self):
        self.sub_menus.add()

    def _add_main_menu(self):
        menu_bar = rt.menuMan.getMainmenuBar()
        self.main_menu = rt.menuMan.createMenu(self.main_menu_name)

        main_menu_item = rt.menuMan.createSubMenuItem(self.main_menu_name, self.main_menu)
        main_menu_id = menu_bar.numItems() + 1
        menu_bar.addItem(main_menu_item, main_menu_id)
