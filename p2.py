# IR Evaluation Plots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# -----------------------------
# Sample documents and query
docs = [
    "information retrieval uses binary independence model",
    "probabilistic models are important in information retrieval",
    "bm25 is derived from the binary independence model",
    "neural networks are widely used for information retrieval"
]
query = "binary independence retrieval"

# Assume ground truth relevance (binary: 1 = relevant, 0 = non-relevant)
# Let's say docs 0 and 2 are relevant
ground_truth = np.array([1, 0, 1, 0])

# -----------------------------
# Step 1: Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
q_vec = vectorizer.transform([query])

# Step 2: Compute similarity
scores = cosine_similarity(q_vec, X).flatten()

# -----------------------------
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(ground_truth, scores)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

# -----------------------------
# ROC Curve
fpr, tpr, _ = roc_curve(ground_truth, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], 'k--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.grid(True)
plt.show()

# -----------------------------
# nDCG Curve
def dcg_at_k(rels, k):
    rels = np.asfarray(rels)[:k]
    if rels.size:
        return np.sum((2**rels - 1) / np.log2(np.arange(2, rels.size + 2)))
    return 0.

def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.

# Relevance values sorted by system ranking
ranking = np.argsort(-scores)
sorted_rels = ground_truth[ranking]

k_values = [1, 2, 3, 4]
ndcg_scores = [ndcg_at_k(sorted_rels, k) for k in k_values]

plt.figure(figsize=(6, 4))
plt.plot(k_values, ndcg_scores, marker='o')
plt.xlabel("k (Top Documents)")
plt.ylabel("nDCG")
plt.title("nDCG Curve")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()



# IR Evaluation Plots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# -----------------------------
# Sample documents and query
docs = [
    "information retrieval uses binary independence model",
    "probabilistic models are important in information retrieval",
    "bm25 is derived from the binary independence model",
    "neural networks are widely used for information retrieval"
]
query = "binary independence retrieval"

# Assume ground truth relevance (binary: 1 = relevant, 0 = non-relevant)
# Let's say docs 0 and 2 are relevant
ground_truth = np.array([1, 0, 1, 0])

# -----------------------------
# Step 1: Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
q_vec = vectorizer.transform([query])

# Step 2: Compute similarity
scores = cosine_similarity(q_vec, X).flatten()

# -----------------------------
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(ground_truth, scores)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

# -----------------------------
# ROC Curve
fpr, tpr, _ = roc_curve(ground_truth, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], 'k--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.grid(True)
plt.show()

# -----------------------------
# nDCG Curve
def dcg_at_k(rels, k):
    rels = np.asfarray(rels)[:k]
    if rels.size:
        return np.sum((2**rels - 1) / np.log2(np.arange(2, rels.size + 2)))
    return 0.

def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.

# Relevance values sorted by system ranking
ranking = np.argsort(-scores)
sorted_rels = ground_truth[ranking]

k_values = [1, 2, 3, 4]
ndcg_scores = [ndcg_at_k(sorted_rels, k) for k in k_values]

plt.figure(figsize=(6, 4))
plt.plot(k_values, ndcg_scores, marker='o')
plt.xlabel("k (Top Documents)")
plt.ylabel("nDCG")
plt.title("nDCG Curve")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()



# BIM with Full IR Evaluation (P@k + MAP)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# -----------------------------
# Sample documents and query
docs = [
    "information retrieval uses binary independence model",
    "probabilistic models are important in information retrieval",
    "bm25 is derived from the binary independence model",
    "neural networks are widely used for information retrieval"
]
query = "binary independence retrieval"

# Ground truth relevance (assume docs 0 and 2 are relevant)
ground_truth = np.array([1, 0, 1, 0])

# -----------------------------
# Step 1: Vectorize (binary presence/absence)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(docs).toarray()
q_vec = vectorizer.transform([query]).toarray()[0]

# -----------------------------
# Step 2: Assume relevance feedback
relevant_docs = [0, 2]
non_relevant_docs = [i for i in range(len(docs)) if i not in relevant_docs]

R = len(relevant_docs)
NR = len(non_relevant_docs)

# Term frequencies
df_rel = np.sum(X[relevant_docs], axis=0)
df_nonrel = np.sum(X[non_relevant_docs], axis=0)

# Probabilities
p = (df_rel + 0.5) / (R + 1)
q = (df_nonrel + 0.5) / (NR + 1)

# RSJ weights
weights = np.log((p * (1 - q)) / (q * (1 - p)))

# -----------------------------
# Step 3: Score documents using BIM formula
scores = X @ (q_vec * weights)

# -----------------------------
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(ground_truth, scores)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("BIM: Precision-Recall Curve")
plt.grid(True)
plt.show()

# -----------------------------
# ROC Curve
fpr, tpr, _ = roc_curve(ground_truth, scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"BIM: ROC Curve (AUC = {roc_auc:.2f})")
plt.grid(True)
plt.show()

# -----------------------------
# nDCG Curve
def dcg_at_k(rels, k):
    rels = np.asfarray(rels)[:k]
    if rels.size:
        return np.sum((2**rels - 1) / np.log2(np.arange(2, rels.size + 2)))
    return 0.

def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.

ranking = np.argsort(-scores)
sorted_rels = ground_truth[ranking]

k_values = [1, 2, 3, 4]
ndcg_scores = [ndcg_at_k(sorted_rels, k) for k in k_values]

plt.figure(figsize=(6, 4))
plt.plot(k_values, ndcg_scores, marker='o')
plt.xlabel("k (Top Documents)")
plt.ylabel("nDCG")
plt.title("BIM: nDCG Curve")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()

# -----------------------------
# Precision@k (P@k)
def precision_at_k(rels, k):
    return np.mean(rels[:k])

p_at_k = [precision_at_k(sorted_rels, k) for k in k_values]

plt.figure(figsize=(6, 4))
plt.bar(k_values, p_at_k)
plt.xlabel("k (Top Documents)")
plt.ylabel("Precision@k")
plt.title("BIM: Precision@k")
plt.ylim(0, 1.05)
plt.grid(True, axis='y')
plt.show()

# -----------------------------
# Mean Average Precision (MAP)
def average_precision(rels):
    precisions = [precision_at_k(rels, k) for k in range(1, len(rels) + 1) if rels[k-1] == 1]
    return np.mean(precisions) if precisions else 0

map_score = average_precision(sorted_rels)

print(f"Mean Average Precision (MAP): {map_score:.4f}")
