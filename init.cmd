@echo off
set "__VENV=.venv"
call %__VENV%\Scripts\activate
set "PATH=%PATH%;%CD%\src\scripts"
pip install -e .