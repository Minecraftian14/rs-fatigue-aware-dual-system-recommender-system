0<0# : ^
'''
@echo off
python "%~f0" %* > temp.cmd & temp & del temp.cmd
'''
import dotenv, os
dotenv.load_dotenv()
model = os.environ['lmstudio.model']
print(f'@REM Specify the model to use with lm studio')
print(f'@set LOCAL_MODEL_API_KEY=""')
print(f'@deepeval set-local-model --model="{model}" --base-url="http://localhost:1234/v1/"')
