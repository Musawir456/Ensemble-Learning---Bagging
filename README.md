# Ensemble Learning - Bagging

A machine learning project implementing ensemble learning techniques, specifically **Bagging (Bootstrap Aggregating)**, to predict health conditions using multiple predictive models. This repository demonstrates the application of ensemble methods for improved accuracy in diabetes and heart disease prediction.

## 📋 Project Overview

This project applies ensemble learning principles to build robust predictive models for medical diagnosis. By combining multiple base learners trained on different subsets of the data (bagging technique), we achieve higher accuracy and reduced variance compared to single models.

**Key Features:**
- Implementation of Bagging ensemble technique
- Diabetes prediction model
- Heart disease prediction model  
- Real-world health datasets
- Comparative analysis of ensemble vs. single models

## 📁 Repository Structure

```
Ensemble-Learning---Bagging/
├── bagging_diabetes_prediction.ipynb      # Diabetes prediction using bagging
├── bagging_heart_disease_prediction.ipynb # Heart disease prediction using bagging
├── heart.csv                              # Heart disease dataset
└── README.md                              # This file
```

## 🎯 Datasets

### Heart Disease Dataset (`heart.csv`)
- Contains cardiovascular health metrics and target diagnosis
- Used for training heart disease prediction models
- Features include age, sex, cholesterol, blood pressure, and other cardiac indicators

### Diabetes Dataset (Implied)
- Used for diabetes prediction model
- Typically includes health metrics relevant to diabetes detection

## 🔧 Technologies & Libraries

- **Python 3.x** - Programming language
- **Jupyter Notebook** - Interactive development environment
- **Scikit-learn** - Machine learning library (decision trees, bagging, ensemble methods)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization

## 📚 What is Bagging?

Bagging (Bootstrap Aggregating) is an ensemble learning technique that:

1. **Creates multiple subsets** of the original dataset through random sampling with replacement (bootstrapping)
2. **Trains base learners** on each bootstrap sample independently
3. **Aggregates predictions** through averaging (regression) or voting (classification)
4. **Reduces variance** while maintaining bias, leading to better generalization

**Advantages:**
- Reduces overfitting
- Improves model stability
- Reduces prediction variance
- Works particularly well with high-variance algorithms (e.g., decision trees)

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn jupyter matplotlib seaborn
```

### Running the Models

1. **Clone the repository**
   ```bash
   git clone https://github.com/Musawir456/Ensemble-Learning---Bagging.git
   cd Ensemble-Learning---Bagging
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Run the notebooks**
   - Open `bagging_diabetes_prediction.ipynb` for diabetes prediction
   - Open `bagging_heart_disease_prediction.ipynb` for heart disease prediction
   - Execute cells sequentially to train and evaluate models

## 📊 Project Workflow

### 1. Data Loading & Exploration
- Load datasets (CSV files)
- Explore data structure, distributions, and missing values
- Perform descriptive statistics

### 2. Data Preprocessing
- Handle missing values
- Feature scaling/normalization
- Train-test split (typically 80-20 or 70-30)

### 3. Model Development
- Train base learners (e.g., Decision Trees)
- Implement Bagging wrapper around base learners
- Configure hyperparameters (n_estimators, max_samples, max_features)

### 4. Model Evaluation
- Calculate performance metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Score
  - Confusion Matrix
- Compare bagging ensemble vs. single model performance
- Cross-validation for robust evaluation

### 5. Results & Visualization
- Plot accuracy comparisons
- Visualize feature importance
- Display confusion matrices
- Generate classification reports

## 📈 Expected Results

The bagging ensemble models typically demonstrate:
- **Higher accuracy** compared to single decision tree models
- **Reduced variance** in predictions across different data splits
- **Better generalization** to unseen data
- **Robust performance** across multiple training iterations

## 🔍 Model Components

### Base Learner
- Primary algorithm used: Decision Trees
- Trained independently on bootstrap samples
- Provides diversity in ensemble

### Bagging Ensemble
- Number of estimators: Configurable (typically 10-100)
- Bootstrap samples: Randomly sampled with replacement
- Aggregation method: Majority voting (classification)

### Feature Set
For **Heart Disease Prediction:**
- Age, Sex, Chest Pain Type
- Blood Pressure, Cholesterol, Fasting Blood Sugar
- Max Heart Rate, Exercise Induced Angina
- ST Depression, Slope, Number of Major Vessels

For **Diabetes Prediction:**
- Pregnancies, Glucose, Blood Pressure
- Skin Thickness, Insulin, BMI
- Diabetes Pedigree Function, Age

## 📝 Usage Examples

### Basic Implementation
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create bagging ensemble
base_model = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(base_model, n_estimators=50, random_state=42)

# Train and evaluate
bagging_model.fit(X_train, y_train)
accuracy = bagging_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

## 🎓 Learning Outcomes

By exploring this project, you'll learn:
- Ensemble learning concepts and advantages
- Bagging technique implementation
- Hyperparameter tuning for ensemble models
- Medical data analysis and prediction
- Model evaluation and comparison techniques
- Best practices in machine learning workflows

## 📖 Additional Resources

- [Scikit-learn Bagging Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [Ensemble Methods in Machine Learning](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Bootstrap Aggregating (Bagging)](https://en.wikipedia.org/wiki/Bootstrap_aggregating)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add new prediction models
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Musawir456**  
GitHub: [@Musawir456](https://github.com/Musawir456)

## 📬 Contact & Support

For questions or suggestions, please open an issue on the GitHub repository.

---
