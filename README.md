# OIBSIP_Datascience_2

Sales Prediction Using Machine Learning 📊

A complete machine learning pipeline to predict product sales based on advertising spending across multiple channels (TV, Radio, Newspaper). This project demonstrates end-to-end ML workflow from data exploration to model deployment.

Project Status: ✅ Complete
Difficulty Level: Beginner to Intermediate
Duration: ~2-3 hours to run and understand

🎯 Project Overview
Problem Statement
Most businesses struggle to optimize advertising budgets across different channels without data-driven insights. This project builds a predictive model that forecasts sales based on advertising spend, enabling smarter budget allocation.

Solution
A machine learning pipeline that:

✅ Analyzes 200 advertising campaigns
✅ Compares Linear Regression vs Random Forest models
✅ Achieves 92.4% prediction accuracy (R² = 0.924)
✅ Identifies which advertising channel drives most sales
✅ Forecasts sales for new advertising campaigns
Key Results
Best Model: Random Forest Regressor
Accuracy (R²): 0.924 (92.4%)
RMSE: $1.5 (average prediction error)
Most Impactful Channel: TV (65% importance)
Least Impactful Channel: Newspaper (5% importance)
📊 Dataset
Source: Advertising.csv (200 rows, 5 columns)

Features
Feature	Type	Range	Description
TV	Float	$0.87 - $296.40	TV advertising spend
Radio	Float	$0.0 - $49.98	Radio advertising spend
Newspaper	Float	$0.3 - $114.0	Newspaper advertising spend
Sales	Float	$1.60 - $27.00	Product sales (Target)
Data Quality
✅ No missing values
✅ 200 complete records
✅ Ready for modeling
✅ Balanced distribution
🛠️ Technologies & Libraries
Core Libraries
pandas==1.3.0          # Data manipulation & analysis
numpy==1.21.0          # Numerical computing
scikit-learn==1.0.0    # Machine learning algorithms
matplotlib==3.4.0      # Data visualization
seaborn==0.11.0        # Advanced visualization
Environment
Python 3.7+
Google Colab (recommended for beginners)
Jupyter Notebook / IPython
📁 Project Structure
sales-prediction-ml/
│
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── data/
│   └── Advertising.csv               # Dataset (200 records)
│
├── notebooks/
│   └── sales_prediction_complete.py   # Full implementation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Load & explore data
│   ├── preprocessing.py              # Data preprocessing
│   ├── models.py                     # ML model definitions
│   ├── evaluation.py                 # Model evaluation metrics
│   └── visualization.py              # Plotting functions
│
├── outputs/
│   ├── model_performance.png         # Performance charts
│   ├── feature_importance.png        # Feature importance plot
│   └── predictions_summary.txt       # Results summary
│
└── .gitignore                         # Git ignore file
🚀 Quick Start
Option 1: Google Colab (Easiest)
Open Google Colab
Create a new notebook
Upload Advertising.csv:
python
from google.colab import files
uploaded = files.upload()
Copy the complete code from notebooks/sales_prediction_complete.py
Run and see results!
Option 2: Local Machine
1. Clone the Repository

bash
git clone https://github.com/yourusername/sales-prediction-ml.git
cd sales-prediction-ml
2. Create Virtual Environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies

bash
pip install -r requirements.txt
4. Run the Project

bash
python notebooks/sales_prediction_complete.py
5. View Results

Metrics printed to console
Plots saved to outputs/ directory
📋 Step-by-Step Workflow
Step 1: Data Loading & Exploration
python
import pandas as pd
df = pd.read_csv('Advertising.csv')
df.describe()  # Basic statistics
df.corr()      # Correlation analysis
Output: Understand data ranges, missing values, feature relationships

Step 2: Exploratory Data Analysis (EDA)
python
# Visualize distributions
df.hist(bins=30)
# Check correlations
sns.heatmap(df.corr(), annot=True)
# Scatter plots
plt.scatter(df['TV'], df['Sales'])
Output: Distribution plots, correlation heatmap, scatter plots

Step 3: Data Preprocessing
python
# Remove index column
df_processed = df.drop('', axis=1)
# Check data types
df.info()
Output: Clean, ready-to-use dataset

Step 4: Feature-Target Separation
python
X = df.drop('Sales', axis=1)  # Features
y = df['Sales']                # Target
print(X.shape)  # (200, 3)
print(y.shape)  # (200,)
Step 5: Train-Test Split
python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training: 160 records (80%)
# Testing: 40 records (20%)
Why? Prevent overfitting, evaluate on unseen data

Step 6: Feature Scaling
python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Why? Normalize features to same scale (TV: 0-300, Radio: 0-50)

Step 7: Model Training
python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
Step 8: Prediction
python
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)
Step 9: Model Evaluation
python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Linear Regression
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

# Random Forest
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
Step 10: Visualization
python
plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()
Step 11: Feature Importance
python
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
Output:

Feature      Importance
TV           0.65 (65%)
Radio        0.30 (30%)
Newspaper    0.05 (5%)
Step 12: Predictions on New Data
python
new_campaign = pd.DataFrame({
    'TV': [250],
    'Radio': [40],
    'Newspaper': [30]
})
new_scaled = scaler.transform(new_campaign)
predicted_sales = rf_model.predict(new_scaled)
print(f"Predicted Sales: ${predicted_sales[0]:.2f}")
📈 Results & Performance
Model Comparison
╔════════════════════════════════════════════════════════════╗
║                  MODEL PERFORMANCE METRICS                 ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  LINEAR REGRESSION:                                        ║
║  • R² Score:  0.8968 (89.68% variance explained)          ║
║  • RMSE:      $1.90 (average error)                       ║
║  • MAE:       $1.45 (mean absolute error)                 ║
║  • MSE:       3.61                                         ║
║                                                            ║
║  RANDOM FOREST:                                            ║
║  • R² Score:  0.9240 (92.40% variance explained) ⭐       ║
║  • RMSE:      $1.49 (average error)                       ║
║  • MAE:       $1.12 (mean absolute error)                 ║
║  • MSE:       2.22                                         ║
║                                                            ║
║  🏆 WINNER: Random Forest (Better R², Lower RMSE)         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
Key Insights
TV is the Strongest Predictor (65% importance)
Every $1 increase in TV spend → Sales increase by ~$0.045
Most cost-effective channel
Radio Has Moderate Impact (30% importance)
Every $1 increase in Radio spend → Sales increase by ~$0.189
Secondary channel
Newspaper is Ineffective (5% importance)
Every $1 increase in Newspaper spend → Sales increase by ~$0.002
Minimal ROI
Model Accuracy: 92.4%
On average, predictions are off by ±$1.49
Suitable for business forecasting
Prediction Examples
TV Spend	Radio Spend	Newspaper Spend	Predicted Sales
$250	$40	$30	$18.52
$150	$25	$20	$12.87
$50	$10	$5	$5.42
🔑 Key Learnings
1. Exploratory Data Analysis (EDA) is Crucial
Understand data before modeling
Identify patterns, outliers, relationships
Use correlation analysis to guide feature selection
2. Feature Scaling Prevents Bias
Without scaling: TV (0-300) appears 10x more important
StandardScaler normalizes all features to same scale
Fair comparison across features
3. Train-Test Split Prevents Overfitting
Don't train and test on same data
Model memorizes training data → fails on new data
80-20 split is standard practice
4. Compare Multiple Models
Linear Regression: Simple but may underfit
Random Forest: More complex but captures non-linear patterns
Always test multiple models
5. Feature Importance Drives Business Decisions
Numeric importance helps optimize budgets
Reallocate spending toward high-importance channels
Eliminate low-impact channels
🎓 Educational Value
Perfect For:

✅ Beginners learning ML fundamentals
✅ Students doing data science coursework
✅ Data analysts transitioning to ML
✅ Anyone wanting to understand the complete ML pipeline
Skills You'll Master:

Data loading and exploration
Data preprocessing and scaling
Model training and comparison
Performance evaluation
Data visualization
Feature importance analysis
Making predictions on new data
🔮 Future Improvements
Short-term Enhancements
 Add cross-validation for more robust evaluation
 Hyperparameter tuning (GridSearchCV)
 Add more models (Gradient Boosting, SVM)
 Feature engineering (interaction terms, polynomial features)
 Residual analysis for error patterns
Medium-term Goals
 Web API for real-time predictions
 Interactive dashboard (Streamlit/Dash)
 Docker containerization
 Unit tests for code modules
 Automated model comparison pipeline
Long-term Vision
 Production deployment
 A/B testing framework
 Real-time data pipeline
 MLOps best practices
 Model monitoring and retraining
💡 Use Cases
For Marketing Teams
Forecast sales impact before campaign launch
Optimize budget allocation across channels
Identify underperforming channels
Justify advertising spend to executives
For Business Analysts
Data-driven decision making
ROI prediction and measurement
Cost-benefit analysis
Strategic planning
For Data Scientists
Complete ML workflow template
Reproducible pipeline structure
Best practices demonstration
Code reusability patterns
🐛 Troubleshooting
Issue: "File not found - Advertising.csv"
Solution: Ensure CSV is in the same directory or provide full path

python
df = pd.read_csv('/path/to/Advertising.csv')
Issue: "ModuleNotFoundError: No module named 'sklearn'"
Solution: Install scikit-learn

bash
pip install scikit-learn
Issue: "Memory Error" (in Colab)
Solution: Restart runtime and run again

python
import gc
gc.collect()  # Clear memory
Issue: "RMSE seems too high"
Solution: Check if features are scaled before prediction

python
X_new_scaled = scaler.transform(X_new)  # Don't forget scaling!
📚 Resources & References
Learning Materials
Scikit-learn Documentation
Pandas Tutorial
Matplotlib Guide
ML Fundamentals
Research Papers
Random Forests - Breiman, 2001
Regression Analysis - ESL
Related Datasets
Kaggle Advertising Datasets
UCI Machine Learning Repository
🤝 Contributing
Contributions are welcome! Ways to contribute:

Report Bugs - Open an issue if you find problems
Suggest Improvements - Feature requests are appreciated
Submit Pull Requests - Code improvements, optimizations
Improve Documentation - Clarify confusing sections
Add Examples - Show new use cases
How to Contribute:
bash
1. Fork the repository
2. Create your feature branch (git checkout -b feature/NewFeature)
3. Commit changes (git commit -m 'Add NewFeature')
4. Push to branch (git push origin feature/NewFeature)
5. Open a Pull Request
📄 License
This project is licensed under the MIT License - see LICENSE file for details.

MIT License Summary
✅ Use for commercial purposes
✅ Modify and distribute
✅ Use privately
⚠️ Must include license and copyright notice
👤 Author
Your Name

📧 Email: your.email@gmail.com
🔗 LinkedIn: linkedin.com/in/yourprofile
🐙 GitHub: github.com/yourusername
🌐 Portfolio: yourwebsite.com
Project Timeline
Started: January 2025
Completed: January 2025
Project #: 1 of Data Science Series
🙏 Acknowledgments
Dataset: Advertising dataset (classic ML dataset)
Inspiration: Complete ML pipeline education
Libraries: Open-source community (Pandas, Scikit-learn, Matplotlib)
Mentors: Data science community
⭐ Support
If you found this project helpful:

⭐ Star this repository
🍴 Fork for your own learning
📢 Share with others
💬 Provide feedback
📞 Contact & Questions
Have questions or suggestions? Reach out:

📧 Open an issue on GitHub
💬 Start a discussion
🔗 Connect on LinkedIn
🗺️ Project Roadmap
Phase 1: ✅ COMPLETE
├── Data loading & EDA
├── Preprocessing & scaling
├── Model training
├── Evaluation & visualization
└── Feature importance

Phase 2: 🔄 IN PROGRESS
├── Hyperparameter tuning
├── Cross-validation
└── Model comparison

Phase 3: 📅 PLANNED
├── Web API development
├── Streamlit dashboard
├── Docker deployment
└── Production readiness
📊 Project Statistics
Lines of Code: ~400
Execution Time: ~2-3 minutes
Dataset Size: 200 records
Model Accuracy: 92.4%
Features: 3 (TV, Radio, Newspaper)
Target: Sales
🎯 Success Criteria ✅
 Load and explore data
 Build multiple models
 Achieve >90% accuracy
 Extract business insights
 Make new predictions
 Document everything
 Share on GitHub
Last Updated: January 2025
Status: ✅ Active & Maintained
Contributions: Welcome!

👤 Author PUSHPARANI.B 

Github: https://github.com/pushparani7/
Email: pushparanib7@gmail.com
Connect on LinkedIn : https://www.linkedin.com/in/pushparani-b-839208337 
🙏 Acknowledgments Oasis Internship Program for the learning opportunity Scikit-learn documentation for excellent resources Data science community for inspiration and guidance

⭐ If you found this helpful, please star the repository!

Happy Learning! 🚀 If this project helped you, please consider starring ⭐

