# Employee Performance Analysis and Prediction

## Repository Outline

```
.
├── asset/
│   ├── modeling-flowchart.jpg               - Visual summary of the end-to-end modeling workflow
│   └── with-RobustScaler.png                - Experimental training result snapshot from the initial RobustScaler
│
├── dataset/
│   ├── dataset-description.png              - Image file containing the original dataset description
│   ├── HRDataset_v12_cleaned.csv            - Cleaned dataset used for modeling preparation
│   └── HRDataset_v14.csv                    - Raw employee performance dataset used as the main source data
│
├── model_saving/
│   ├── gb_best_model.pkl                    - Saved best-performing Gradient Boosting classification model
│   ├── list_num_cols.txt                    - List of numerical features used in the final modeling pipeline
│   └── list_cat_cols.txt                    - List of categorical features used in the final modeling pipeline
│
├── 1_employee-performance_analysis.ipynb    - Notebook for exploratory data analysis and business insight generation
├── 2_employee-performance_modeling.ipynb    - Notebook for preprocessing, model training, and evaluation
├── 3_employee-performance_inference.ipynb   - Notebook for loading the saved model and running prediction inference
├── README.md                                - Project documentation and summary of the end-to-end workflow
└── LICENSE                                  - License information for this repository
```

## Overview

This project explores employee performance through an end-to-end workflow, starting from exploratory analysis and continuing into multiclass classification modelling.

The main objective is to understand which employee conditions, work patterns, and organizational signals are most closely associated with stronger or weaker performance outcomes, then translate those findings into a predictive workflow that can support HR evaluation more consistently.

The project is designed as a decision-support tool rather than a replacement for human judgment. In an HR setting, model outputs should be used to highlight patterns, support earlier review, and help prioritize follow-up, especially for employees who may show signs of weaker performance.

Model Deployment Demo: [Hugging Face](https://huggingface.co/spaces/seankafka/Employee-Performance-Prediction)

## Key Insight

Employee performance in this dataset appears to be more closely related to **day-to-day work behaviour, engagement, satisfaction, and employment context** than to static personal background.

That makes the final model most useful as an **early warning and review-support tool**, especially for detecting employees who may fall into `Needs Improvement` or `PIP`. Its value is stronger at identifying potential risk on the lower end of performance than at making sharp distinctions between `Exceeds` and `Fully Meets`.

## Objectives

The goals of this project are:

* Understand how `PerformanceScore` is distributed and identify any class imbalance that may affect modelling
* Explore which employee-related signals are most closely associated with stronger or weaker performance
* Build a modelling-ready dataset through cleaning, feature engineering, and careful feature selection
* Compare multiple baseline machine learning models under the same preprocessing workflow
* Tune the strongest baseline model and evaluate it on an untouched test set
* Save the final model and supporting metadata for reuse in inference and deployment

## Dataset

This project uses the **HRDataset v14** dataset authored by **Dr. Richard Huebner** and published on **Kaggle**.

The dataset contains **311 employee records** and initially includes **36 columns** covering:

* Demographics and employee profile
* Employment history and review records
* Compensation and benefits
* Engagement and satisfaction
* Attendance-related behaviour
* Department and employment context

Target variable:

* **`PerformanceScore`**

Target distribution:

* **Fully Meets**: 243
* **Exceeds**: 37
* **Needs Improvement**: 18
* **PIP**: 13

This distribution shows that lower-performance classes are relatively small, which makes class imbalance an important issue throughout the modelling stage.

## End-to-End Workflow

### 1. Data Cleaning and Preparation

The project begins by cleaning redundant columns, dropping identifier-like variables, checking duplicates, reviewing missing values, trimming whitespace issues, and investigating zero-variance or overly noisy fields.

At this stage, several paired encoded columns are removed in favour of more interpretable business-facing columns. Raw address fields are also excluded because they behave more like identifiers than meaningful performance drivers.

### 2. Exploratory Data Analysis

The analysis notebook focuses on understanding how `PerformanceScore` behaves across numeric, categorical, and derived datetime-based features.

Two derived features, `StartWorkAge` and `Tenure`, are created to test whether employee timing-related background adds useful context. The notebook then compares performance against variables such as salary, engagement, satisfaction, tardiness, absences, special-project exposure, employment status, department, and manager structure.

### 3. Feature Engineering and Selection

For the modelling stage, the feature set is revised further so the workflow remains compact, interpretable, and suitable for HR use.

The final input data for training uses **13 features**, consisting of:

* **8 numeric features**: `Salary`, `EngagementSurvey`, `EmpSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, `StartWorkAge`, `Tenure`
* **5 categorical features**: `Sex`, `EmploymentStatus`, `Department`, `IsMarried`, `RecruitmentGroup`

### 4. Preprocessing and Imbalance Handling

Because the dataset contains both numeric and categorical variables, preprocessing is handled in multiple stages.

The final pipeline applies:

* `StandardScaler` for numeric features
* `SMOTENC` for imbalance handling while preserving categorical structure
* `OneHotEncoder` for the remaining categorical variables after resampling

This order is important because categorical values need to remain intact when `SMOTENC` creates synthetic minority samples.

### 5. Baseline Model Comparison

Five baseline models are compared using the same preprocessing structure:

* K-Nearest Neighbors
* Support Vector Machine
* Decision Tree
* Random Forest
* Gradient Boosting

Because the target is imbalanced, the comparison focuses on **macro precision**, **macro recall**, and **macro F1-score** under 5-fold cross-validation rather than relying on accuracy alone.

### 6. Hyperparameter Tuning and Final Evaluation

`GradientBoosting` provides the strongest baseline result and is selected for further tuning through `GridSearchCV`.

Upon the test set, the tuned model achieves:

| Evaluation Stage | Metric | Score |
|---|---|---:|
| Cross-validation | Best macro F1-score | 0.6341 |
| Test set | Accuracy | 0.83 |
| Test set | Macro precision | 0.74 |
| Test set | Macro recall | 0.66 |
| Test set | Macro F1-score | 0.69 |

The final result shows very strong performance on `Fully Meets`, while `Needs Improvement` and `PIP` are still captured at a useful level despite their limited class sizes.

## Key Findings

### 1. Performance Issues Exist, but They Are Concentrated

Most employees fall into `Fully Meets`, while `Needs Improvement` and `PIP` account for much smaller groups. This suggests the company is not facing a broad performance collapse, but it does contain a smaller subset of employees who show noticeably weaker signals.

### 2. Day-to-Day Behaviour and Employee Experience Are the Strongest Signals

The clearest separation comes from `DaysLateLast30`, `EngagementSurvey`, and `EmpSatisfaction`.

Lower-performing employees tend to be more frequently late, less engaged, and less satisfied. These variables consistently stand out more than static background variables such as age at hire or tenure.

### 3. Employment Context Adds More Value Than Personal Profile

`EmploymentStatus` provides the strongest categorical signal in the modelling stage. Cause-based termination contains a heavier share of lower-performing employees, while voluntary termination is still largely dominated by `Fully Meets`.

By contrast, variables such as gender, marital status, and manager identity show much weaker standalone separation.

### 4. Salary and Special Projects Act More as Supporting Signals

Lower-performing employees also tend to have lower salary levels and less exposure to special projects, but these variables do not separate classes as strongly on their own as engagement, satisfaction, and lateness.

This suggests they still add context, but they are not the most direct signals of weaker performance in this dataset.

### 5. The Strongest Modelling Result Comes From Sequential Ensemble Learning

The baseline comparison shows that `GradientBoosting` delivers the most balanced overall result. This makes sense because employee performance is likely shaped by multiple interacting signals rather than a single simple rule.

A boosting-based model can capture that structure more effectively by learning step by step and correcting earlier mistakes across many trees.

## Business Recommendations

### 1. Build Early Monitoring Around Behavioural and Experience Signals

HR can monitor patterns such as repeated lateness, lower engagement, and lower satisfaction more closely, because these variables separate performance groups more clearly than most other features.

### 2. Prioritize Follow-Up for Lower-Performance Predictions

Predictions of `Needs Improvement` and `PIP` can be used to trigger earlier review, coaching, or closer manager follow-up before problems become more costly.

### 3. Combine Model Output With Human Review

The model should support structured evaluation, not replace it. Prediction results should be reviewed alongside manager input, recent context, and qualitative evidence before any action is taken.

### 4. Continue Improving Data Coverage for Minority Classes

Additional examples for lower-performance groups would likely improve the model further, especially because class imbalance remains one of the main constraints in this project.

## Limitations

* Statistical associations do not imply causation
* Lower-performance classes are small, which limits class balance and model stability
* `Exceeds` is still harder to distinguish from `Fully Meets`
* Model output should be treated as decision support, not automated judgment
* Some variables may become more or less useful if the dataset grows or the organizational context changes

## Tools and Libraries

* **Language**: Python
* **Data Handling**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Statistical Testing**: SciPy
* **Machine Learning**: scikit-learn, imbalanced-learn
* **Model Saving**: joblib, JSON
* **Deployment Demo**: Hugging Face Spaces

## Getting Started

### Prerequisites

* Python 3.9+
* Jupyter Notebook or JupyterLab
* pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/seankafka/employee-performance-analysis.git
   cd employee-performance-analysis
   ```

2. Install the core dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn joblib notebook
   ```

### Project Order

Run the project in this order:

`1_employee-performance_analysis.ipynb`  
`2_employee-performance_modeling.ipynb`  
`3_employee-performance_inference.ipynb`  

## Author

This project was developed as part of a data learning journey, with a focus on building practical, interpretable, and business-relevant HR analytics skills.

Sean Kafka Adhyaksa  
[GitHub](https://github.com/seankafka) || [LinkedIn](https://www.linkedin.com/in/seankafka/)
