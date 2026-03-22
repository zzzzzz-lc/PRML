import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

file_path = "data.xlsx"
train_data = pd.read_excel(file_path, sheet_name=0)
test_data = pd.read_excel(file_path, sheet_name=1)
X_train, y_train = train_data.iloc[:, 0].values.reshape(-1, 1), train_data.iloc[:, 1].values
X_test, y_test = test_data.iloc[:, 0].values.reshape(-1, 1), test_data.iloc[:, 1].values

def plot_train_test_scatter(X_train, y_train, X_test, y_test, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.xlabel('X_train'), plt.ylabel('y_train'), plt.title('Training Set')
    plt.legend(), plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, color='red', alpha=0.6, label='Testing Data')
    plt.xlabel('X_test'), plt.ylabel('y_test'), plt.title('Testing Set')
    plt.legend(), plt.grid(True, alpha=0.3)

    plt.tight_layout(), plt.savefig(save_path, dpi=300, bbox_inches='tight'), plt.close()

# 绘制训练/测试集散点图
plot_train_test_scatter(X_train, y_train, X_test, y_test, '训练集测试集散点图.png')

def plot_linear_fitting(X_train, y_train, X_test, y_test, w, b,
                        train_mse, test_mse, train_r2, test_r2, method_name, save_path):
    # 线性拟合结果绘图
    plt.figure(figsize=(12, 5))
    # 训练集拟合
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_plot = b + w * X_plot.flatten()
    plt.plot(X_plot, y_plot, color='orange', linewidth=2,
             label=f'Fitting Line: y={w:.4f}x+{b:.4f}')
    plt.xlabel('X_train'), plt.ylabel('y_train')
    plt.title(f'{method_name}-Training Set\nMSE={train_mse:.4f}, R²={train_r2:.4f}')
    plt.legend(), plt.grid(True, alpha=0.3)

    # 测试集拟合
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, color='red', alpha=0.6, label='Testing Data')
    X_plot = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_plot = b + w * X_plot.flatten()
    plt.plot(X_plot, y_plot, color='orange', linewidth=2,
             label=f'Fitting Line: y={w:.4f}x+{b:.4f}')
    plt.xlabel('X_test'), plt.ylabel('y_test')
    plt.title(f'{method_name}-Testing Set\nMSE={test_mse:.4f}, R²={test_r2:.4f}')
    plt.legend(), plt.grid(True, alpha=0.3)

    plt.tight_layout(), plt.savefig(save_path, dpi=300, bbox_inches='tight'), plt.close()


def convergence_curve_error_analysis(loss_history, train_mse, test_mse, train_r2, test_r2,
                                 method_name, save_path):
    # 收敛曲线与误差对比
    plt.figure(figsize=(12, 5))
    # 收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss_history)), loss_history, color='green' if method_name == 'GD' else 'red', linewidth=1.5)
    plt.xlabel('迭代次数'), plt.ylabel('均方误差损失（MSE/2）')
    plt.title(f'{method_name}收敛曲线'), plt.grid(alpha=0.3)

    # 误差对比
    plt.subplot(1, 2, 2)
    metrics = ['MSE', 'R²']
    train_vals = [train_mse, train_r2]
    test_vals = [test_mse, test_r2]
    x_pos = np.arange(len(metrics))
    width = 0.35
    plt.bar(x_pos - width / 2, train_vals, width, label='训练集', color='skyblue', alpha=0.8)
    plt.bar(x_pos + width / 2, test_vals, width, label='测试集', color='lightcoral', alpha=0.8)
    for i, (t1, t2) in enumerate(zip(train_vals, test_vals)):
        plt.text(i - width / 2, t1 + 0.01, f'{t1:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width / 2, t2 + 0.01, f'{t2:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('评价指标'), plt.ylabel('指标值')
    plt.title(f'{method_name}-训练/测试集误差对比')
    plt.xticks(x_pos, metrics), plt.legend(), plt.grid(alpha=0.3, axis='y')

    plt.tight_layout(), plt.savefig(save_path, dpi=300, bbox_inches='tight'), plt.close()


def plot_kernel_ridge_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred,
                              pipeline, train_mse, test_mse, train_r2, test_r2, save_path):
    plt.figure(figsize=(15, 10))
    # 训练集拟合
    plt.subplot(2, 2, 1)
    X_smooth = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
    y_smooth = pipeline.predict(X_smooth)
    plt.scatter(X_train, y_train, color='steelblue', alpha=0.6, s=30, label='训练数据')
    plt.plot(X_smooth, y_smooth, color='crimson', linewidth=2.5, label='RBF核岭回归拟合')
    plt.fill_between(X_smooth.flatten(), y_smooth - np.sqrt(train_mse),
                     y_smooth + np.sqrt(train_mse), alpha=0.2, color='crimson', label='误差范围')
    plt.xlabel('X_train'), plt.ylabel('y_train')
    plt.title(f'训练集拟合\nR²={train_r2:.4f}, MSE={train_mse:.4f}', fontweight='bold')
    plt.legend(), plt.grid(True, alpha=0.3)

    # 测试集拟合
    plt.subplot(2, 2, 2)
    X_smooth = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
    y_smooth = pipeline.predict(X_smooth)
    plt.scatter(X_test, y_test, color='crimson', alpha=0.6, s=30, label='测试数据')
    plt.plot(X_smooth, y_smooth, color='crimson', linewidth=2.5, label='RBF核岭回归拟合')
    plt.fill_between(X_smooth.flatten(), y_smooth - np.sqrt(test_mse),
                     y_smooth + np.sqrt(test_mse), alpha=0.2, color='crimson', label='误差范围')
    plt.xlabel('X_test'), plt.ylabel('y_test')
    plt.title(f'测试集拟合\nR²={test_r2:.4f}, MSE={test_mse:.4f}', fontweight='bold')
    plt.legend(), plt.grid(True, alpha=0.3)

    # 残差分析
    plt.subplot(2, 2, 3)
    train_res = y_train - y_train_pred
    test_res = y_test - y_test_pred
    plt.scatter(X_train, train_res, color='steelblue', alpha=0.6, s=20, label='训练集残差')
    plt.scatter(X_test, test_res, color='crimson', alpha=0.6, s=20, label='测试集残差')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.xlabel('X'), plt.ylabel('残差（y_true - y_pred）')
    plt.title('残差分布分析', fontweight='bold')
    plt.legend(), plt.grid(True, alpha=0.3)

    # 预测值/真实值对比
    plt.subplot(2, 2, 4)
    plt.scatter(y_train, y_train_pred, color='steelblue', alpha=0.6, s=20, label='训练集')
    plt.scatter(y_test, y_test_pred, color='crimson', alpha=0.6, s=20, label='测试集')
    y_range = np.linspace(min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max()), 100)
    plt.plot(y_range, y_range, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='理想拟合线')
    plt.xlabel('真实值 y_true'), plt.ylabel('预测值 y_pred')
    plt.title(f'预测值/真实值对比\nR²={test_r2:.4f}', fontweight='bold')
    plt.legend(), plt.grid(True, alpha=0.3)

    plt.tight_layout(), plt.savefig(save_path, dpi=300, bbox_inches='tight'), plt.close()


def plot_fourier_fitting(X_train, y_train, X_test, y_test,train_mse, test_mse,
                         train_r2, test_r2, n_harmonics, save_path, model, omega):  # 新增：复用训练好的模型和参数
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    X_train_smooth = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)

    X_train_smooth_fourier = np.zeros((len(X_train_smooth), 1 + 2 * n_harmonics))
    X_train_smooth_flat = X_train_smooth.flatten()
    X_train_smooth_fourier[:, 0] = 1.0
    for n in range(1, n_harmonics + 1):
        X_train_smooth_fourier[:, 2*(n-1)+1] = np.sin(n * omega * X_train_smooth_flat)
        X_train_smooth_fourier[:, 2*(n-1)+2] = np.cos(n * omega * X_train_smooth_flat)

    y_train_smooth = model.predict(X_train_smooth_fourier)

    plt.scatter(X_train, y_train, color='steelblue', alpha=0.6, s=30, label='训练数据')
    plt.plot(X_train_smooth, y_train_smooth, color='darkorange', linewidth=2.5,
             label=f'{n_harmonics}次谐波拟合')
    plt.fill_between(X_train_smooth_flat,
                     y_train_smooth - np.sqrt(train_mse),
                     y_train_smooth + np.sqrt(train_mse),
                     alpha=0.2, color='darkorange', label='误差范围')
    plt.xlabel('X_train'), plt.ylabel('y_train')
    plt.title(f'傅里叶拟合-训练集\nR²={train_r2:.4f}, MSE={train_mse:.4f}', fontweight='bold')
    plt.legend(), plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    X_test_smooth = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
    X_test_smooth_fourier = np.zeros((len(X_test_smooth), 1 + 2 * n_harmonics))
    X_test_smooth_flat = X_test_smooth.flatten()
    X_test_smooth_fourier[:, 0] = 1.0
    for n in range(1, n_harmonics + 1):
        X_test_smooth_fourier[:, 2*(n-1)+1] = np.sin(n * omega * X_test_smooth_flat)
        X_test_smooth_fourier[:, 2*(n-1)+2] = np.cos(n * omega * X_test_smooth_flat)
    y_test_smooth = model.predict(X_test_smooth_fourier)

    plt.scatter(X_test, y_test, color='crimson', alpha=0.6, s=30, label='测试数据')
    plt.plot(X_test_smooth, y_test_smooth, color='darkorange', linewidth=2.5,
             label=f'{n_harmonics}次谐波拟合')
    plt.fill_between(X_test_smooth_flat,
                     y_test_smooth - np.sqrt(test_mse),
                     y_test_smooth + np.sqrt(test_mse),
                     alpha=0.2, color='darkorange', label='误差范围')
    plt.xlabel('X_test'), plt.ylabel('y_test')
    plt.title(f'傅里叶拟合-测试集\nR²={test_r2:.4f}, MSE={test_mse:.4f}', fontweight='bold')
    plt.legend(), plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def least_squares_regression(X_train, y_train, X_test, y_test):
    print("最小二乘法线性回归")

    X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
    b, w = theta[0], theta[1]
    print(f"线性结果：y = {w:.6f}*X + {b:.6f}")

    y_train_pred = b + w * X_train.flatten()
    y_test_pred = b + w * X_test.flatten()
    train_mse, test_mse = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
    train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)

    print(f"训练集MSE：{train_mse:.6f}，R²：{train_r2:.6f}")
    print(f"测试集MSE：{test_mse:.6f}，R²：{test_r2:.6f}")

    plot_linear_fitting(X_train, y_train, X_test, y_test, w, b,
                        train_mse, test_mse, train_r2, test_r2,
                        '最小二乘法', '最小二乘法线性拟合直线.png')
    return b, w, train_mse, test_mse, train_r2, test_r2, y_train_pred, y_test_pred


def gradient_descent_regression(X_train, y_train, X_test, y_test, lr=0.01, epochs=5000, tol=1e-6):
    print("\n梯度下降法线性回归")

    # 梯度下降核心类
    class GradientDescent:
        def __init__(self, lr, max_epochs, tol):
            self.b, self.w = np.random.randn(), np.random.randn()
            self.lr, self.max_epochs, self.tol = lr, max_epochs, tol
            self.loss_history = []

        def fit(self, X, y):
            X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            for epoch in range(self.max_epochs):
                old_b, old_w = self.b, self.w
                y_pred = X_bias @ np.array([self.b, self.w])
                self.loss_history.append(np.mean((y_pred - y) ** 2) / 2)
                # 梯度更新
                grad_b, grad_w = np.mean(y_pred - y), np.mean((y_pred - y) * X_bias[:, 1])
                self.b -= self.lr * grad_b
                self.w -= self.lr * grad_w
                if np.sqrt((self.b - old_b) ** 2 + (self.w - old_w) ** 2) < self.tol:
                    print(f"在第{epoch + 1}轮收敛"),
                    break
            if epoch == self.max_epochs - 1: print("达到最大迭代次数")

        def predict(self, X):
            return self.b + self.w * X.flatten()

    gd_model = GradientDescent(lr, epochs, tol)
    gd_model.fit(X_train, y_train)
    y_train_pred, y_test_pred = gd_model.predict(X_train), gd_model.predict(X_test)

    train_mse, test_mse = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
    train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)

    print(f"线性结果：y = {gd_model.w:.6f}*X + {gd_model.b:.6f}")
    print(f"训练集MSE：{train_mse:.6f}，R²：{train_r2:.6f}")
    print(f"测试集MSE：{test_mse:.6f}，R²：{test_r2:.6f}")

    plot_linear_fitting(X_train, y_train, X_test, y_test, gd_model.w, gd_model.b,
                        train_mse, test_mse, train_r2, test_r2,
                        'GD', '梯度下降法线性拟合_拟合直线.png')
    convergence_curve_error_analysis(gd_model.loss_history, train_mse, test_mse, train_r2, test_r2,
                                 'GD', '梯度下降法线性拟合_收敛曲线与误差对比.png')
    return (gd_model.b, gd_model.w, train_mse, test_mse, train_r2, test_r2,
            y_train_pred, y_test_pred, gd_model.loss_history)


def newton_method_regression(X_train, y_train, X_test, y_test, epochs=100, tol=1e-6):
    print("\n牛顿法线性回归")

    # 牛顿法核心类
    class NewtonMethod:
        def __init__(self, max_epochs, tol):
            self.b, self.w = np.random.randn(), np.random.randn()
            self.max_epochs, self.tol = max_epochs, tol
            self.loss_history = []
            self.hessian_eps = 1e-6

        def fit(self, X, y):
            X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            for epoch in range(self.max_epochs):
                old_params = np.array([self.b, self.w])
                y_pred = X_bias @ old_params
                self.loss_history.append(np.mean((y_pred - y) ** 2) / 2)
                # 梯度与海森矩阵
                error = y_pred - y
                grad = np.array([np.mean(error), np.mean(error * X_bias[:, 1])]).reshape(-1, 1)
                hessian = np.array([[1, np.mean(X_bias[:, 1])],
                                    [np.mean(X_bias[:, 1]), np.mean(X_bias[:, 1] ** 2)]]) + np.eye(2) * self.hessian_eps

                delta = np.linalg.inv(hessian) @ grad
                self.b, self.w = self.b - delta[0, 0], self.w - delta[1, 0]
                if np.linalg.norm(np.array([self.b, self.w]) - old_params) < self.tol:
                    print(f"在第{epoch + 1}轮收敛"),
                    break
            if epoch == self.max_epochs - 1: print("达到最大迭代次数")

        def predict(self, X):
            return self.b + self.w * X.flatten()

    newton_model = NewtonMethod(epochs, tol)
    newton_model.fit(X_train, y_train)
    y_train_pred, y_test_pred = newton_model.predict(X_train), newton_model.predict(X_test)

    train_mse, test_mse = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
    train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)

    print(f"线性结果：y = {newton_model.w:.6f}*X + {newton_model.b:.6f}")
    print(f"训练集MSE：{train_mse:.6f}，R²：{train_r2:.6f}")
    print(f"测试集MSE：{test_mse:.6f}，R²：{test_r2:.6f}")

    plot_linear_fitting(X_train, y_train, X_test, y_test, newton_model.w, newton_model.b,
                        train_mse, test_mse, train_r2, test_r2,
                        'Newton', '牛顿法线性拟合_拟合直线.png')
    convergence_curve_error_analysis(newton_model.loss_history, train_mse, test_mse, train_r2, test_r2,
                                 '牛顿法', '牛顿法线性拟合_收敛曲线与误差对比.png')
    return (newton_model.b, newton_model.w, train_mse, test_mse, train_r2, test_r2,
            y_train_pred, y_test_pred, newton_model.loss_history)


def fourier_series_fitting(X_train, y_train, X_test, y_test, n_harmonics=5, alpha=0.005):
    print("\n傅里叶级数拟合")
    period = (X_train.max() - X_train.min()) * 2
    omega = 2 * np.pi / period
    print(f"周期T={period:.2f}，基频ω={omega:.4f}")

    def generate_fourier_features(X, n_harmonics, omega):
        X_flat = X.flatten()
        n_samples = len(X_flat)
        features = np.zeros((n_samples, 1 + 2 * n_harmonics))
        features[:, 0] = 1.0
        for n in range(1, n_harmonics + 1):
            features[:, 2 * (n - 1) + 1] = np.sin(n * omega * X_flat)
            features[:, 2 * (n - 1) + 2] = np.cos(n * omega * X_flat)
        return features

    X_train_fourier = generate_fourier_features(X_train, n_harmonics, omega)
    X_test_fourier = generate_fourier_features(X_test, n_harmonics, omega)

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=False))
    model.fit(X_train_fourier, y_train)

    y_train_pred = model.predict(X_train_fourier)
    y_test_pred = model.predict(X_test_fourier)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"谐波次数：{n_harmonics}，正则化系数：{alpha}")
    print(f"训练集MSE：{train_mse:.6f}，R²：{train_r2:.6f}")
    print(f"测试集MSE：{test_mse:.6f}，R²：{test_r2:.6f}")

    plot_fourier_fitting(X_train, y_train, X_test, y_test, train_mse, test_mse,
                         train_r2, test_r2, n_harmonics,'傅里叶级数拟合结果.png', model, omega)

    return model, n_harmonics, train_mse, test_mse, train_r2, test_r2, y_train_pred, y_test_pred

def kernel_ridge(X_train, y_train, X_test, y_test):
    print("\n高斯核岭回归")
    best_params = {'kernel': 'rbf', 'gamma': 6.0, 'alpha': 0.001}
    pipeline = make_pipeline(StandardScaler(), KernelRidge(**best_params))
    pipeline.fit(X_train, y_train)

    y_train_pred, y_test_pred = pipeline.predict(X_train), pipeline.predict(X_test)
    train_mse, test_mse = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
    train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)

    print(f"核函数：{best_params['kernel']}，gamma：{best_params['gamma']}，alpha：{best_params['alpha']}")
    print(f"训练集MSE：{train_mse:.6f}，R²：{train_r2:.6f}")
    print(f"测试集MSE：{test_mse:.6f}，R²：{test_r2:.6f}")

    plot_kernel_ridge_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred,
                              pipeline, train_mse, test_mse, train_r2, test_r2, '核岭回归结果.png')
    return pipeline, train_mse, test_mse, train_r2, test_r2, y_train_pred, y_test_pred

# 执行各拟合方法
ls_results = least_squares_regression(X_train, y_train, X_test, y_test)
gd_results = gradient_descent_regression(X_train, y_train, X_test, y_test)
newton_results = newton_method_regression(X_train, y_train, X_test, y_test)
fourier_results = fourier_series_fitting(X_train, y_train, X_test, y_test, n_harmonics=5)
kr_results = kernel_ridge(X_train, y_train, X_test, y_test)
