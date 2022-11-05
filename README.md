# Mini-project IV

## Project/Goals
For this project, I wanted to focus on building, grid-searching, and testing an entire pipeline along with integrating Docker into my AWS instance.

## Hypothesis
I will be testing two main hypotheses:
1. That gender plays a significant role in the determination of a bank loan
2. That credit score plays a significant role in the determination of a bank loan

## Process

### EDA

[EDA Notebook](notebooks/DateExploration.ipynb)

After giving the data an intial look to get a first impression, I looked through the data to find NaN values, audit the entries, and examined outliers. Looking at the histograms of each feature I found that a number of features were heavily skewed which may introduce bias into our dataset. Also, the applicant and coapplicant income are positively skewed so we will need to take a log transformation.

Looking at the correlation matrix of our feature we see a positive correlation between credit history and positive loan approval. There is a small difference with gender but it isn't immediately clear from the plot how much of an influence it has.

![Correlation Matrix](images/corr_plot.png)

### Proof of Concept Pipeline
[Simple Model Notebook](notebooks/Simple_Model.ipynb)

After preforming my EDA, I created an extremely simple pipeline that allowed me to "fail fast and fail early." I wanted to make sure I had a proof of concept regarding pipelines and flask before moving on with the project. To do this, I did the following:

  1. Dropped all non-quantitative features from the dataset
  2. Dropped all NaN rows
  3. Created, fit, and predicted a pipeline: SimpleScaler -> RandomForestClassifier -> GridSearchCV (only searching between two params)
  4. Saved resulting model as a pickle file
  5. Uploaded the pickle along with a basic flask API to AWS
  
Having created this basic pipeline, I felt ready to move on to the project proper.

### Pipeline Building: Cleaning & Feature Engineering


### Modelling


### Deployment


### Testing


## Results


### Model

### API

## Challanges 
(discuss challenges you faced in the project)

## Future Goals
(what would you do if you had more time? are there any potential issues/biases with your model/use case?)
