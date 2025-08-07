# Pokemon-Legendary-Prediction
* This repository presents a supervised learning approach to classify Pokémon as Legendary or Not Legendary using statistical and categorical features from the dataset. Multiple classification models (K-Nearest Neighbors, Logistic Regression, Random Forest, XGBoost) were evaluated, with a focus on model performance and feature interpretability. https://www.kaggle.com/datasets/shreyasdasari7/golden-pokdex-kanto-to-hoenn

  ## OVERVIEW

  * **Background:** Legendary Pokémon are rare and powerful creatures in the Pokémon universe. Understanding what distinguishes them from regular Pokémon based on stats and attributes can be a fun and insightful classification problem.
  * **Project Goal:** To build a classification model that predicts whether a Pokémon is Legendary based on its stats, type, and other metadata.
  * **Approach:** This is a supervised binary classification task. After preprocessing and feature engineering, we trained several models and evaluated them using metrics like Accuracy, Precision, Recall, F1 Score, and ROC AUC. Feature importance analysis was performed using XGBoost.
  * **Summary of Performance** The XGBoost classifier outperformed other models and achieved the highest overall balance between precision and recall. Feature importance visualization showed which attributes were most predictive.

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Pokémon dataset including features like HP, Attack, Defense, Speed, Type1, Type2, Generation, Capture Rate, etc.
  * **Target:**
    * Legendary – whether a Pokémon is legendary (1) or not (0)
  * **Size:**
    * ~800 Pokémon records, each with 15+ features

#### Preprocessing / Clean up

* **Missing Values:**
  * Removed or filled NaN values in Type2, Capture Rate, and others
* **Dropped Columns:**
  * Removed Total Base Stats, Catch Method, Number, and other biased/redundant columns
* **One-Hot Encoding:**
  * Converted categorical features such as Type1, Type2, and Color into binary variables using pd.get_dummies
* **Normalization:**
  * Scaled numerical features using StandardScaler for consistent model input
* **Other:**
  * Checked and preserved high stat outliers like Mewtwo, as they represent real Legendary traits

#### Data Visualization & Performance Comparison

<img width="539" height="154" alt="Screenshot 2025-08-07 at 10 49 04 AM" src="https://github.com/user-attachments/assets/e92f9721-d4c2-4dad-9c38-b98296875716" />

Each model was assessed using five metrics: Accuracy, Precision, Recall, F1 Score, and ROC AUC.

The results show that XGBoost achieved the highest overall performance, with near-perfect scores in all categories. It maintained a strong Recall of 0.67 and F1 Score of 0.80, while also achieving a ROC AUC of 1.00, meaning it perfectly separated Legendary from non-Legendary Pokémon in terms of prediction probability.
Random Forest also performed well in terms of Accuracy and ROC AUC, but its Recall (0.33) was lower—indicating it missed more Legendary Pokémon compared to XGBoost. Logistic Regression showed decent overall performance with a balanced Recall and AUC, but weaker Precision, suggesting more false positives.

In contrast, KNN failed to identify any Legendary Pokémon, resulting in zero values for Precision, Recall, and F1 Score, and a low ROC AUC of 0.418. This highlights that KNN is not a suitable model for this imbalanced classification task.

Below is the corresponding ROC Curve, which visualizes the trade-off between the true positive rate and false positive rate at various thresholds for each model:

The ROC curve reinforces the metrics: both XGBoost and Random Forest follow the top-left edge of the plot (ideal classifier), confirming their excellent discriminatory power. Logistic Regression follows closely, while KNN lies closer to the diagonal line, indicating poor separation between classes.

<img width="1038" height="786" alt="Screenshot 2025-08-07 at 10 48 43 AM" src="https://github.com/user-attachments/assets/a62cdbe1-7c99-4e29-b53b-dc17a339fe7a" />

Key Observations:
XGBoost and Random Forest both achieved perfect or near-perfect classification, with an AUC (Area Under Curve) of 1.00, indicating excellent performance.
Logistic Regression performed very well, with an AUC of 0.95, showing it is a strong baseline model.
KNN (k=5) performed poorly in comparison, with an AUC of 0.42, which is worse than random guessing (0.50), likely due to its sensitivity to class imbalance or lack of distinct clusters in the feature space.
The diagonal dashed line represents random performance (i.e., flipping a coin), so models with curves closer to the top-left corner perform better.
This chart helps us visualize and compare model discrimination ability and supports our decision to favor tree-based ensemble models like Random Forest and XGBoost for this classification task.

<img width="925" height="447" alt="Screenshot 2025-08-07 at 10 54 16 AM" src="https://github.com/user-attachments/assets/640ec8a8-7a4a-480e-a79e-f4f6d44b10cb" />

To gain further insight into how the model makes predictions, we analyzed the feature importances generated by the XGBoost classifier. The chart above displays the top 10 most influential features based on their importance scores within the model.
Base HP stands out as the most influential predictor of a Pokémon’s Legendary status, with a significantly higher importance score (~3.73) compared to other features. This suggests that Legendary Pokémon are often distinguished by unusually high HP values, which aligns with their expected in-game strength and durability.

Following Base HP, other key predictors include Base Friendship, Base Speed, and Base Special Attack, all of which scored above 1.5. These attributes highlight characteristics that may be indicative of Legendary Pokémon—such as strong combat abilities and stat advantages that are typically not found in regular Pokémon.

Interestingly, features like Height (m) and Weight (kg), while included, showed lower importance scores (below 0.5), indicating that physical characteristics are less useful in determining whether a Pokémon is Legendary compared to battle-related stats.

By understanding which features most strongly influence model decisions, we not only enhance the interpretability of our results but also gain domain-specific insights into what traits set Legendary Pokémon apart from the rest.


### Problem Formulation

* **Input/Output**
  * Input: A set of Pokémon attributes including base stats (e.g., HP, Attack, Speed), physical features (e.g., Height, Weight), and other metadata (e.g., Base Friendship).
  * Output: A binary label indicating whether the Pokémon is Legendary (1) or Not Legendary (0).
* **Models Used:**
  To evaluate the performance of different classification techniques, the following models were implemented and compared:
  * **Logistic Regression:** A baseline linear classifier used to model the relationship between Pokémon features and their Legendary status.
  * **K-Nearest Neighbors (KNN):** A non-parametric model based on feature proximity. While simple, it struggled with sparse class imbalance and high-dimensional data.
  * **Random Forest:** An ensemble of decision trees that improves performance by reducing overfitting and capturing non-linear feature interactions.
  * **XGBoost:** A gradient boosting framework known for its performance on structured data. It achieved the highest overall predictive performance and was further analyzed for feature importance.
* **Loss Function & Evaluation Metrics**
  As a binary classification task, all models were optimized to minimize binary crossentropy loss where applicable (e.g., Logistic Regression, XGBoost).
Model performance was assessed using:
  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC AUC Score
  These metrics were selected to provide a comprehensive view of performance, especially under class imbalance (Legendary Pokémon being the minority class).
* **Hyperparamters Tuning & Class Imbalance Handling**
  * Class Weighting: Applied to penalize misclassification of the minority class (Legendary Pokémon), improving recall and F1-Score.
  * Threshold Tuning: The classification threshold was adjusted to balance between precision and recall, typically targeting higher recall to ensure more Legendary Pokémon were correctly identified.
  * Train-Test Split: Stratified splitting ensured both training and testing sets preserved the original class distribution.


### Training

Model training was conducted in Jupyter Notebook on a MacBook Pro equipped with an Apple M1 chip and 8 GB of memory. The training pipeline made use of several key Python libraries and frameworks:
* scikit-learn: Used extensively for data preprocessing (e.g., one-hot encoding, normalization), model training (Logistic Regression, KNN, Random Forest), and evaluation metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC).
* XGBoost: Employed for training a gradient-boosted decision tree classifier, which ultimately delivered the strongest predictive performance in this project.
* Matplotlib: Used for data visualization, including ROC curves and feature importance plots.
* Pandas/NumPy: Used for data wrangling, exploration, and feature engineering.
Each model trained in under 10 seconds due to the relatively small dataset size and tabular nature of the features. XGBoost took slightly longer due to its iterative boosting process, but still completed within seconds. The entire training and evaluation process took approximately 2 hours, including data cleaning, tuning, and result visualization.

**Challenges:** Early in the training process, a key challenge was overfitting—especially with models like Random Forest and XGBoost, which can easily memorize small datasets. This was mitigated by limiting tree depth and using stratified train-test splits to preserve class balance.

Another challenge was class imbalance: Legendary Pokémon make up a small fraction of the dataset. This issue was addressed by applying class weights and focusing on recall and ROC AUC rather than accuracy alone, ensuring the model prioritized identifying rare Legendary Pokémon over predicting the majority class.

A mistake made early on was evaluating model thresholds directly on the test set, which led to inflated performance metrics. This was corrected by refining the workflow to tune thresholds on the training or validation split before final testing. This change led to more honest and reliable performance reporting.


### Conclusions

After comparing four different classification models—Logistic Regression, K-Nearest Neighbors (k=5), Random Forest, and XGBoost—we determined that XGBoost offered the best balance of performance across all key metrics. With an Accuracy of 0.98, Precision of 1.0, Recall of 0.67, F1 Score of 0.8, and a perfect ROC AUC score of 1.0, XGBoost consistently outperformed the other models in distinguishing Legendary Pokémon from non-Legendary ones.
To further understand how the model made its predictions, we visualized the top 10 most important features according to XGBoost. Notably, attributes like Base HP, Base Friendship, and Base Speed were among the most influential, suggesting that Legendary status may be partially defined by high durability and strong affinity metrics. This aligns with in-universe expectations, where Legendary Pokémon are generally more powerful and distinct in both base stats and lore.

While simpler models like Logistic Regression showed decent results, they fell short in precision and overall discriminatory power. KNN struggled significantly due to the high dimensionality and sparse nature of the dataset after encoding, which led to poor generalization. On the other hand, ensemble tree-based methods like Random Forest and XGBoost effectively captured non-linear relationships and feature interactions that boosted prediction accuracy.

A key takeaway is that feature engineering and model selection both played critical roles in achieving strong performance. By excluding bias-inducing columns and normalizing input data, we improved the generalizability of the models. Additionally, careful evaluation using ROC curves and class-specific metrics allowed us to select a model that minimizes false positives while maintaining a high detection rate for rare Legendary cases.

Overall, this project demonstrates the feasibility and value of using machine learning to predict Pokémon rarity based on statistical attributes. It also highlights the importance of balancing accuracy with recall and precision—especially in cases where class imbalance could bias results. With further enhancements, such as more granular type analysis or evolutionary lineage data, this model could be extended to support advanced Pokémon scouting tools or game analytics systems.


### Future Work

* **Explore Advanced Feature Engineering**
  * While the current model performs well using base stats and simple metadata, future iterations could incorporate type effectiveness, evolution stage, egg group, or legendary lore references from external data sources (e.g., Pokémon lore or game version exclusives). Additionally, interaction terms between stats (e.g., Attack × Speed) may uncover deeper predictors of Legendary status.
* **Test Additional Algorithms and Neural Architectures**
  * Future work could involve trying out neural networks, LightGBM, or even AutoML pipelines to compare their performance to XGBoost. While XGBoost performed exceptionally in this project, deep learning models could potentially capture subtler nonlinear patterns if more complex features are introduced.
* **Build an Interactive Pokémon Classification Tool**
  * Create a web-based dashboard or app where users can input Pokémon stats and metadata to receive a real-time prediction: “Is this Pokémon likely to be Legendary?” This could include dynamic feature visualizations, confidence scores, and comparisons to known Legendaries.
* **Apply Transfer Learning to Pokémon from New Generations**
  * As new generations of Pokémon are introduced, the model could be retrained or fine-tuned using transfer learning techniques to generalize across new data while retaining learned patterns from previous generations.
 

 ## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The list below outlines the main components of this project in the order they were developed and utilized:
* **Tabular Project Proposal.pdf:** PDF version of the Tabualr Project Proposal.pptx powerpoint
* **Young Tabular Feasibility.ipynb:** This notebook contains the exploratory data analysis (EDA), data preprocessing, and feature engineering. It helped identify redundant or biased features (e.g., Total Base Stats) and shaped the final cleaned dataset.
* **YoungTabularPrototype2.ipynb:** An improved and final prototype that includes visualizations of ROC curves, confusion matrices, and feature importance. It also integrates final metric tuning and selection of the most suitable model.
* **Hoenn.csv, Johto.csv, Kanto.csv:** Regional subsets of Pokémon data, possibly used for regional performance comparison or additional data validation. These were not part of the final model training but were explored for potential expansion or breakdowns.

### Software Setup

This project was developed and executed entirely within Jupyter Notebook, using the Anaconda distribution as the development environment:

* **Data Handling & Visualization:**
  * pandas, numpy, matplotlib, seaborn
* **Preprocessing & Evaluation:**
  * sklearn.preprocessing:
    * MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
  * sklearn.model_selection:
    * train_test_split
  * sklearn.metrics:
    * ccuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
* **Machine Learning Models:**
  * sklearn.linear_model.LogisticRegression
  * sklearn.neighbors.KNeighborsClassifier
  * sklearn.ensemble.RandomForestClassifier
  * xgboost.XGBClassifier
 
### Training

* Install Required Packages: All required packages were installed within a Jupyter Notebook environment using pip. The essential libraries included pandas, numpy, scikit-learn, matplotlib, seaborn, and xgboost. This ensured compatibility across all classifiers used in the project.
* Data Preparation: The dataset was preprocessed using the notebook Young Tabular Feasibility.ipynb. This included:
    * Dropping biased or redundant features (e.g., Total Stats, Catch Method)
    * Normalizing numeric columns using StandardScaler
    * Encoding categorical features using LabelEncoder and OneHotEncoder
    * Splitting the dataset into training and test sets (80/20 split)
    * Addressing class imbalance between Legendary and non-Legendary Pokémon
* Model Training Order: The following models were trained sequentially in YoungTabularPrototype.ipynb and YoungTabularPrototype2.ipynb:
    * Logistic Regression
    * K-Nearest Neighbors (K=5)
    * Random Forest
    * XGBoost
* Training Configuration: Although deep learning was not used in this project, each model was trained using scikit-learn or XGBoost’s implementation with:
    * train_test_split for hold-out evaluation
    * 5-fold cross-validation where applicable for robustness
    * Metrics like Accuracy, Precision, Recall, F1-score, and ROC AUC for evaluation
* Threshold Tuning: For models like XGBoost and Random Forest, predicted probabilities were analyzed on the validation set to determine an optimal decision threshold. Instead of relying solely on a default 0.5 threshold, threshold tuning was used to balance Recall and F1 Score, especially important given the class imbalance in predicting Legendary status.
* Final Evaluation on Test Set: Once optimal thresholds were established, they were applied to the test set. Final performance was assessed using a combination of metrics and visualizations like ROC curves, confusion matrices, and feature importance rankings.


## CITATIONS
https://www.kaggle.com/datasets/shreyasdasari7/golden-pokdex-kanto-to-hoenn
Chatgpt assisted in helping solve errors that occured.
