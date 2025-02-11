python -m venv .venv
.\.venv\Scripts\pip3 install --upgrade pip
.\.venv\Scripts\pip3 install -r .\requirements.txt
call .\.venv\Scripts\activate.bat
echo Press key to open Visual Studio Code
timeout /t 30
code .