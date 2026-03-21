@echo off
cd src\report
pandoc -t markdown_strict --extract-media=../../attachments main.tex -o ../../README.md
cd ..\..