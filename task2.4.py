import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False



# 数据生成函数
def make_moons_3d(n_samples=500, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y


# 构建模型

def build_models():
    models = {}

    # 未优化基线模型
    models["Based Decision Tree"] = DecisionTreeClassifier(
        max_depth=5,
        random_state=2026
    )

    stump_baseline = DecisionTreeClassifier(max_depth=2, random_state=2026)
    models["Based AdaBoost + DT"] = AdaBoostClassifier(
        estimator=stump_baseline,
        n_estimators=80,
        learning_rate=0.3,
        random_state=2026,
    )

    models["Based SVM (Linear)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=1.0, random_state=2026)),
    ])

    models["Based SVM (Poly)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="poly",
            degree=2,
            C=3.0,
            gamma="scale",
            coef0=0.5,
            random_state=2026
        )),
    ])

    models["Based SVM (RBF)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=5.0,
            gamma=0.5,
            random_state=2026
        )),
    ])

    models["Based SVM (Sigmoid)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="sigmoid",
            C=1.0,
            gamma="scale",
            coef0=1.0,
            random_state=2026
        )),
    ])

    # 优化后模型
    models["Optimized Decision Tree"] = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=2026
    )

    stump_optimized = DecisionTreeClassifier(max_depth=3, random_state=2026)
    models["Optimized AdaBoost + DT"] = AdaBoostClassifier(
        estimator=stump_optimized,
        n_estimators=200,
        learning_rate=1.0,
        random_state=2026,
    )

    models["Optimized SVM (Poly)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="poly",
            degree=3,
            C=1.0,
            gamma="scale",
            coef0=1.0,
            random_state=2026
        )),
    ])

    models["Optimized SVM (RBF)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            random_state=2026
        )),
    ])

    models["Optimized SVM (Sigmoid)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="sigmoid",
            C=5.0,
            gamma=0.05,
            coef0=-0.5,
            random_state=2026
        )),
    ])

    return models

# 模型训练
def evaluate(models, X_train, y_train, X_test, y_test):
    rows = []
    predictions = {}

    for name, model in models.items():
        print(f"正在训练: {name}")
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        rows.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, train_pred),
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
        })

    result = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)
    return result, predictions

# 4. 绘制性能对比柱状图

def plot_comparison(result):
    # 分离基线和优化模型
    baseline_mask = result["Model"].str.contains("Based")
    optimized_mask = result["Model"].str.contains("Optimized")

    baseline = result[baseline_mask].copy()
    optimized = result[optimized_mask].copy()

    # 提取模型名（去掉前缀）
    baseline["Model"] = baseline["Model"].str.replace("Based", "")
    optimized["Model"] = optimized["Model"].str.replace("Optimized", "")

    # 合并数据
    comparison = pd.merge(
        baseline[["Model", "Test Accuracy"]],
        optimized[["Model", "Test Accuracy"]],
        on="Model",
        suffixes=(" Based", " Optimized")
    )

    # 画图
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(comparison))
    width = 0.35

    bars1 = ax.bar(x - width / 2, comparison["Test Accuracy Based"], width, label="未优化基线", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, comparison["Test Accuracy Optimized"], width, label="优化后", color="#ff7f0e")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("测试集准确率", fontsize=12)
    ax.set_title("模型优化前后性能对比", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Model"], rotation=30, ha="right")
    ax.set_ylim([0.45, 1.0])
    ax.legend()

    # 在柱子上标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{height:.3f}",
                    ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("optimization_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# 最佳模型分类结果
def plot_best_model_3d(X, y_true, y_pred, model_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    correct = y_true == y_pred
    wrong = ~correct

    X_correct = X[correct]
    y_correct = y_true[correct]

    scatter = ax.scatter(X_correct[:, 0], X_correct[:, 1], X_correct[:, 2],
                         c=y_correct,
                         cmap='viridis',
                         marker='o',
                         s=30,
                         alpha=0.7)

    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    if np.any(wrong):  # 如果有错误分类的点
        ax.scatter(X[wrong, 0], X[wrong, 1], X[wrong, 2],
                   c='red',
                   marker='x',
                   s=120,
                   linewidth=2,
                   label='Wrong Prediction')
        ax.legend(loc='upper right')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'3D Classification Result\nBest Model: {model_name}')

    plt.tight_layout()
    plt.savefig("best_model_3d_classification.png", dpi=300, bbox_inches="tight")
    plt.show()


X_train, y_train = make_moons_3d(n_samples=500, noise=0.2, random_state=2026)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2, random_state=2027)

models = build_models()

result, predictions = evaluate(models, X_train, y_train, X_test, y_test)

result.to_csv("optimization_results.csv", index=False)

# 绘制对比图
plot_comparison(result)

# 最佳模型3D分类图
best_model_name = result.iloc[0]['Model']
best_y_pred = predictions[best_model_name]
plot_best_model_3d(X_test, y_test, best_y_pred, best_model_name)