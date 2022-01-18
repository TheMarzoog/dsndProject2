# Disaster Response Pipeline Project
TO DO: add project summery


### Installation:
1. Download the ZIP file of the project or type: `git clone https://github.com/marzoogtech/dsndProject2.git` on the terminal to clone the project.
2. Install the necessary libraries numpy, pandas, sklearn using: pip install <library-name> on the terminal or other methods.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Descriptions:
1. data:
    - disaster_messages.csv: contains messages from the disaster
    - disaster_categories.csv: contains categories of the disaster
    - DisasterResponse.db: contains cleaned data
2. models:  
    - classifier.pkl: contains trained model
    - classifier.py: contains ML pipeline
3. app:
    - run.py: contains web app
    - templates: contains html templates

