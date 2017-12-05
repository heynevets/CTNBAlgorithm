This project is based on python 3.6.3

Please follow the instruction below to run the program

---------------Hardware Requirement---------------
This project requires around 6 GBs of free hardisk space to store the data


--------------------------------------------------



1. Download the following data from https://www.kaggle.com/stackoverflow/stacksample/data
    Answers.csv
    Tags.csv
    Questions.csv

2. execute command:
    pip install -r requirement.txt

3. execute command: 
    python DataParsing.py
    # This script will parse the necessary data to the /archive folder year by year

4. execute command:
    python B1_GenerateLDAModel.py
    # This script will generate LDA model and store it in the /LDAModel folder
    # The script will also generate year vs topic scoring matrix and documents count for each topic
    # Line 65 and 66 can be modified to change the number of topics and number of training passes, respectively


