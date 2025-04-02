# Phoenix Pham (3037848920)

# How to reproduce the Homework 5 results:
1. Run all cells of the ```tree_depth_analysis.ipynb``` file on Jupyter notebook. Ensure that your Python environment is configured properly with required dependencies.

# How to reproduce the Kaggle results:
1. Run the ```decision_tree_starter.py``` script to generate the results for the Titanic and Spam datasets. To choose which dataset to predict, ensure you set the `dataset` variable in the main function to the desired dataset. For the Titanic dataset, I used RandomSearchCV to hyperparameter tune the parameters for the Random Forest classifier, so ensure the `pred` variable at the end of the script uses the `best_rf` model. I didn't hyperparamter tune the Spam dataset, so ensure `pred` uses the original `rf` model.
