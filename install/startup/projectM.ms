-- 스크립트 파일 경로 
scriptPath = getFilenamePath(getThisScriptFilename())

-- 설치 파일을 실행
python.ExecuteFile (scriptPath + "projectM.py")