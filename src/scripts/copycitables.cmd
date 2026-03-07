@setlocal
@echo off
cd %~dp0
cd ..\..\planner\survey
grep -Pro "status: \w+ Citation" | grep -v "File Not Available" | cut -d / -f 2- | cut -d . -f -1 | clip
powershell get-clipboard
endlocal
