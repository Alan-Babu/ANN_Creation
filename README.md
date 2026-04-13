## create Environment
conda create -p <env name> python==<specific version>
Eg: conda create -p venv python==3.13

## Activate Env
conda activate <env name>
Eg: conda activate venv
## Install the dependencies
pip install -r ./requirements.txt

## Run the app 
streamlit run ./app.py

