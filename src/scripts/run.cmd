@setlocal
@echo off
call compilereferences
cp src\report\references.bib build\report\temp\references.bib
latexmk
@endlocal
