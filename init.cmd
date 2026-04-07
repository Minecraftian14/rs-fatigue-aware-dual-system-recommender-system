@echo off
set "__VENV=.venv_cpu"
call %__VENV%\Scripts\activate
set "PATH=%PATH%;%CD%\src\scripts"
pip install -e .