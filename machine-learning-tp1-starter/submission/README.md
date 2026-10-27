# ML TP1: Python for Data Exploration and Preprocessing

Everything is in `submission/tp1_solution.ipynb`. The data files are in `data/`.
Run the notebook from top to bottom (Restart and Run All).

Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib (see `requirements.txt`).

---

## Exercise 1: Iris dataset

**What each inspection output tells me**
- `info()`: the column names, the type of each column and how many non-null values it has. Good for spotting missing values and wrong types.
- `describe()`: mean, std, min, quartiles and max of the numeric columns. It gives a quick idea of the scale and spread of each feature.
- `shape`: number of rows and columns.
- `columns`: the list of column names.
- `head()`: the first 5 rows, to see what the data really looks like.

**Why make a copy?** So I can try preprocessing without damaging the original data. If something goes wrong, I still have the untouched DataFrame.

**Features and target:** the four numeric measurements (sepal and petal sizes) are the input features. The species column is the target.

**Split:** 80% train and 20% test, `random_state=42` so the result is the same every time. I used `stratify=y` so each species has the same proportion in both sets.

**Discussion: why keep the test set separate until the end?**
The test set is supposed to act like new data the model has never seen. If its information leaks into preprocessing (for example computing the mean for scaling on all the data) or into model selection (picking the model that scores best on it), the final score looks better than it really is. This is data leakage. The model would then perform worse on real new data than the test score suggests.

---

## Exercise 2: Text data

**Dimensions of 20 Newsgroups (train subset):** the number of documents is how many texts there are. The number of labels is one label per document, so both numbers are the same.

**Target and class names:** `target` contains integers. Each integer is the index of a class in `target_names`. For example, if the target of a document is 7, its class name is `target_names[7]`.

---

## Exercise 3: Missing values

**Why are '-' and 'na' not counted as missing?** Pandas only treats `NaN`, `None` and `NaT` as missing. '-' and 'na' are just normal strings, so `isna()` ignores them. I had to convert them to `np.nan` myself.

**Conversion rule:** a cell is replaced by `np.nan` only if it is exactly '-' or 'na' (any letter case). This way I don't accidentally change real values that only contain those characters.

**Imputation strategy:** median for numerical columns, most frequent value for categorical columns.

**Limitations of imputation:**
- The filled values are guesses, not real data.
- It reduces the variance of the data and can distort distributions and correlations.
- It can add bias if the data is not missing at random.
- When building a real model, the imputer must be fitted on the training set only, then used to transform the validation and test sets.

---

## Exercise 4: Scaling and encoding

### Part A: Scaling and binarization

| Method | Works on | Idea |
|---|---|---|
| StandardScaler | each feature | subtract the mean, divide by the std (mean 0, std 1) |
| MinMaxScaler | each feature | rescale to the range [0, 1] |
| MaxAbsScaler | each feature | divide by the max absolute value (range [-1, 1]), keeps zeros as zeros |
| RobustScaler | each feature | uses the median and the IQR, so outliers affect it much less |
| normalize | each sample (row) | scales each row so its norm is 1 |

**Binarization:** with a threshold of 0, values greater than 0 become 1 and the others become 0. A higher threshold gives more zeros, a lower one gives more ones.

**Discussion: does scaling matter?**
Yes for k-NN and SVM because they use distances: a feature with big values would dominate the others. Gradient-based models also converge faster and more stably when features have similar scales. Decision trees don't need it, because each split looks at one feature at a time and only depends on the order of the values, not on their scale.

### Part B: Encoding categorical variables

**OrdinalEncoder:** it gives each category a number (alphabetical order, see `categories_`). These numbers are only codes, not a real ranking.

**Target encoding:** `LabelEncoder` gives one integer per class, and I printed the mapping between class names and numbers.

**Integer vs one-hot:** integer encoding uses a single column and creates an order between classes. One-hot uses one binary column per class, with a 1 in the column of the right class and 0 elsewhere, so no order is implied.

**Why one-hot for nominal inputs?** Features like continent or browser have no natural order. With ordinal codes (for example Europe = 1, US = 2), a model could think US is "greater" than Europe, which makes no sense. One-hot avoids this false order.

---

## Conclusion

- **Numerical data:** handle missing values (median), then scale or normalize if the algorithm needs it. Binarization is possible when only a threshold matters.
- **Categorical data:** handle missing values (most frequent), then encode. Use one-hot for nominal input features, and label/ordinal encoding for targets.
- **Text data:** read the file with the right encoding (UTF-8), then turn the text into numbers (word counts or TF-IDF) before modeling.
- In every case: fit the preprocessing on the training set only, then apply it to the test set, to avoid data leakage.