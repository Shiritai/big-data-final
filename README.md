# Final Project - Big Data

## Set Up

```
conda create --name your_env_name python=3.10
conda activate your_env_name
pip install pandas
```

## Submission Format

Please provide a zip file with this structure:

```
r119020XX/
|- main.py # your code for clustering
|- r119020XX_public.csv # for evaluation on the public dataset
|- r119020XX_private.csv # for evaluation on the private dataset
|_ r119020XX_report.pdf



```

Each file should follow the format below:

```
id,label
0,1
1,2
2,3
...
```

#### id should follow the original sample order. label is your predicted value.

## Validation

Use eval.py to check your performance on the public dataset:
`python eval.py`
