@echo off
REM Tek tuşla main_tr.tex derleme script'i (Windows, PowerShell gerekmez)
cd /d "%~dp0"
pdflatex main_tr.tex
bibtex main_tr
pdflatex main_tr.tex
pdflatex main_tr.tex
echo.
echo Derleme tamamlandi. Olusan PDF: main_tr.pdf
pause

