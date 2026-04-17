@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Virtual environment created and activated!
echo You can now run the data curating script.
cmd /k
