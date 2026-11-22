{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ce48f0b2-a87a-4e60-9109-160d86f242e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import confusion_matrix, recall_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f5a95907-6a0a-465d-b67d-77f8fcba3d97",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('cancer.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4a0f009-cb27-49d2-806a-8e4a0c4f4cba",
   "metadata": {},
   "source": [
    "# Breast Cancer Prediction using Logistic Regression\n",
    "\n",
    "**Goal:** Predict whether a tumor is malignant (1) or not (0).\n",
    "**Target column:** `Class`\n",
    "**Model:** Logistic Regression with StandardScaler (scikit-learn pipeline)\n",
    "**Result:** Test accuracy 95.6%, precision 0.98, recall 0.91"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a9b90782-044b-41d7-84d6-32801433a989",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy of my model is  0.9562043795620438\n",
      "error rate 0.04379562043795615\n",
      "precision my model  0.9811320754716981\n",
      "f1_score  0.9454545454545454\n",
      "confuse_martics  [[79  1]\n",
      " [ 5 52]]\n",
      "recall  0.9122807017543859\n",
      "Train: 0.9706422018348624\n",
      "Test: 0.9562043795620438\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.99      0.96        80\n",
      "           1       0.98      0.91      0.95        57\n",
      "\n",
      "    accuracy                           0.96       137\n",
      "   macro avg       0.96      0.95      0.95       137\n",
      "weighted avg       0.96      0.96      0.96       137\n",
      "\n"
     ]
    }
   ],
   "source": [
    "X=data.drop(columns=['Class'])\n",
    "y=data['Class']\n",
    "\n",
    "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)\n",
    "\n",
    "model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\n",
    "model.fit(X_train,y_train)\n",
    "y_predict=model.predict(X_test)\n",
    "accuracy=accuracy_score(y_test,y_predict)\n",
    "print('accuracy of my model is ',accuracy)\n",
    "print('error rate',1-accuracy)\n",
    "print('precision my model ',precision_score(y_test,y_predict))\n",
    "print('f1_score ',f1_score(y_test,y_predict))\n",
    "print('confuse_martics ',confusion_matrix(y_test,y_predict))\n",
    "print('recall ',recall_score(y_test,y_predict))\n",
    "print(\"Train:\", model.score(X_train, y_train))\n",
    "print(\"Test:\", model.score(X_test, y_test))\n",
    "print(classification_report(y_test, y_predict))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98e375c0-2c2b-44ef-94a6-422f43314c31",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73c23559-cdae-427f-a3bc-e23b62f03f54",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
