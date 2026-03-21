@setlocal
@echo off
cd %~dp0
cd ..\..\planner\survey
grep -r link | sed -E "s/^[a-zA-Z0-9_ -]*\///" | sed "s/.md:link//" | clip
powershell get-clipboard
endlocal