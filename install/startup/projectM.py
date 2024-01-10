import sys  
from pymxs import runtime as rt
path = rt.GetDir(rt.Name('userStartupScripts')) + '\\'
if path not in sys.path:  
    sys.path.append(path)
    
import ncMaxPipeline as ncm
ncm.install_projectM_menu()