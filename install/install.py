import os
import shutil
import stat

from pymxs import runtime as rt


class Install:

    def __init__(self):
        self.set_variables()
        self.copy_project()
        self.copy_startup_script()
        self.open_path(self.script_path)

    def copy_startup_script(self):
        """startup script�� �����Ѵ�."""
        self.delete_old_start_up_script()
        shutil.copy(self.shared_start_up_script_path,
                    self.local_start_up_script_path)

    def delete_old_start_up_script(self):
        if self.exists_path(self.local_start_up_script_path):
            self.delete_path(self.local_start_up_script_path)

    def set_variables(self):
        """ȯ�� �������� �����Ѵ�."""
        self.project_name = 'ncMaxPipeline'
        self.start_up_script_file = 'projectM.ms'
        self.shared_path = '\\\\imartsh\\share\\ART\\TA\\ScriptShare\\'
        self.script_path = rt.GetDir(rt.Name('userStartupScripts')) + '\\'

        self.shared_project_path = self.shared_path + self.project_name
        self.local_project_path = self.script_path + self.project_name
        self.shared_start_up_script_path = self.shared_project_path + '\\install\\startup' + \
                                           self.start_up_script_file
        self.local_start_up_script_path = self.script_path + self.start_up_script_file

    def copy_project(self):
        """���������� �ִ� ������Ʈ�� �����Ѵ�.
        
        Notes:
            �����Ҷ��� ncMaxPipeline/ncMaxPipeline ������ �����ϴ� �Ŵ�.
        """
        if not self.exists_path(self.shared_project_path):
            raise RuntimeError(f"Shared path is not exists. {self.shared_project_path}")
        self.delete_old_project()
        shutil.copytree(self.shared_project_path + '\\' + self.project_name,
                        self.local_project_path)

    def delete_old_project(self):
        """���ÿ� ����ִ� ���� ������Ʈ�� �����Ѵ�."""
        if self.exists_path(self.local_project_path):
            self.delete_path(self.local_project_path)

    def open_path(self, path):
        """��θ� �����ش�."""
        if self.exists_path(path):
            os.startfile(path.replace('/', '\\'))

    def exists_path(self, path):
        """��ΰ� �����ϴ��� Ȯ���Ѵ�."""
        if path:
            return os.path.exists(path)
        else:
            return False

    def delete_path(self, path):
        """���� ���� �ƹ��ų� ������ִ�.

        ������ ������ ����°� ������ ���� ������ Ȯ���ϰ� ���� ������ ���ߴ�.
        """
        if '.' in path:
            os.chmod(path, stat.S_IWRITE)
            os.remove(path)
        else:
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    Install()
