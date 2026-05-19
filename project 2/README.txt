# 🤖 Iris Flower Classifier – KNN Supervised Learning
# Project 2 | DecodeLabs Batch 2026
# Data Classification Using AI
# ============================================================
#
# HOW TO RUN:
# 1. Install required libraries:
#    pip install scikit-learn matplotlib seaborn pandas numpy
# 2. Run this file:
#    python iris_classifier.py
# 3. View results in terminal output
#
# ============================================================
# WHAT THIS PROJECT DOES:
#
# - Loads the classic Iris dataset (150 flowers, 3 species, 4 features)
# - Scales features using StandardScaler (mean=0, variance=1)
# - Splits data: 80% training (120 samples), 20% testing (30 samples)
# - Trains a K-Nearest Neighbors model with K=5
# - Predicts species on unseen test data
# - Evaluates with Confusion Matrix and F1 Score