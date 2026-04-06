@echo off
start cmd /k "py main.py"
start cmd /k "cd oncology-dashboard && npm run dev"
echo Servers started in new windows.
