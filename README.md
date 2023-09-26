# Explanation
This repo is used to showcase results, thought process and methodologies used to tackle the task given by `CAKE`.

# Dependencies
All the needed dependencies are in `requirements.txt` and are easily installed. Navigate to the project folder using the following command in the Terminal `pip install -r "requirements.txt"`.

# Structure
Due to the fact that this task has limited time, we will somewhat of a simple structure. Structure is as follows:
- `artifacts`: folder where the model will be exported and imported from
- `data`: folder where the data if generated from running the notebooks as well as the data for training
- `notebooks`: folder where we keep notebooks to execute the whole process (step by step) 
- `testing model`: folder where we will simulate process of calling the endpoint (in this case we will just import the model and make predictions)

# Future work
All of the work could be set to run on cloud (namely `Azure ML`, `AWS Sagemaker`), which would reduce time for data preparation, feature engineering, model training. That would also help the model become more scalable in case new data arrives.
