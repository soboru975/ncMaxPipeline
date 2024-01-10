# max의 unserStartupScript에서 해당내용을 부른다.
import sys  
path = "D:\\projects\\scripts\\ncMaxPipeline\\"  
if path not in sys.path:  
    sys.path.append(path)