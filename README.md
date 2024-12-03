# Customer Segmentation Using Neural Networks and SMOTE

## Problem Statement
Businesses need accurate methods to classify and segment customers to target marketing efforts effectively and maximize revenue. By predicting customer segments based on spending patterns, engagement, and other behavioral metrics, businesses can refine their strategies to retain high-value customers and attract new ones. This project builds a machine-learning model to classify customers into segments such as "Low Value," "Medium Value," and "High Value" using behavioral data.

---

## Background Info
Customer segmentation is a core strategy in business analytics, but it often suffers from class imbalance. High-value customers are typically underrepresented in datasets, making it challenging to build accurate models that generalize well. Techniques such as SMOTE (Synthetic Minority Oversampling Technique) and neural networks with robust hyperparameter tuning are employed to address these challenges.

---

## Methodology
1. **Simulate Business Data:** Created a realistic dataset based on spending, engagement, and other key metrics.
2. **Preprocessing:**
   - Addressed class imbalance with SMOTE.
   - Scaled numerical features using `StandardScaler`.
   - Encoded categorical variables with one-hot encoding.
3. **EDA:** Visualized data distributions and class imbalance before and after SMOTE.
4. **Neural Network Training:** Built a multi-class classification model using TensorFlow/Keras.
5. **Hyperparameter Optimization:** Tuned the neural network using Keras Tuner.
6. **Evaluation:** Measured performance with metrics like accuracy, precision, recall, and F1-score.

---

## Data Information

### Initial Dataset
- **Rows:** 1,500 (simulated)
- **Columns:** 6 features + 1 target (Customer Segment)
- **Class Distribution (Before SMOTE):**
  - Low Value: 1,067
  - Medium Value: 393
  - High Value: 100

### Final Dataset (After SMOTE)
- **Rows:** 1,920 (balanced across classes)
- **Columns:** 6 features + 1 target

---

## Data Dictionary
| Feature               | Type  | Description                                      |
|-----------------------|-------|--------------------------------------------------|
| **Monthly Spending**  | float | Average monthly spending by the customer         |
| **Engagement Score**  | int   | Interaction score (range: 1–100)                 |
| **Tenure**            | int   | Duration as a customer (months)                  |
| **Complaint Count**   | int   | Number of complaints filed (Poisson distributed) |
| **Purchase Frequency**| float | Average purchases per month                      |
| **Discount Usage**    | float | Fraction of purchases made with discounts (0–1) |
| **Customer Segment**  | int   | Target variable: Low (0), Medium (1), High (2)   |

---

## EDA

### Class Distribution
#### Before SMOTE
- Low Value: Dominates with 1,067 samples.
- Medium Value: Moderately represented with 393 samples.
- High Value: Severely underrepresented with only 100 samples.

#### After SMOTE
- All classes balanced with 640 samples each.

### Visualizations
- **Bar Charts:** Showed class distribution before and after SMOTE.
- **Box Plots:** Highlighted feature distributions across segments.

---

## Summary of Analysis

### Key Observations
- Class imbalance severely impacted initial model performance.
- SMOTE balanced the dataset, improving minority class recall.

### Model Performance
- **Best Validation Accuracy:** 98.39%
- **Test Accuracy:** 96.15%
- High precision and recall across all classes, including the minority class (High Value).

---

## Conclusions and Recommendations

### Conclusions
- The tuned neural network achieved excellent classification accuracy and generalization.
- SMOTE successfully balanced the dataset, ensuring equitable performance across all customer segments.

### Recommendations
- Deploy the model in marketing and customer retention applications.
- Regularly retrain with updated data to maintain accuracy and adaptability.

---

## What's Next?

### Model Improvements
- Experiment with alternative architectures (e.g., convolutional layers for feature extraction).
- Implement ensemble methods for further accuracy improvement.

### Dataset Expansion
- Include new features like location, age, and product preferences.
- Gather real-world data for validation.

### Application Development
- Build a user-friendly interface for customer segmentation visualization.
- Integrate with CRM tools for real-time insights.

---

## Sources
1. [TensorFlow Documentation](https://www.tensorflow.org/)
2. [Keras Tuner Guide](https://keras.io/keras_tuner/)
3. SMOTE Paper: Chawla, N.V. et al. "SMOTE: Synthetic Minority Over-sampling Technique." 2002.
