#获得所有文件
dates_list = []
data_df_list = []
year = int(input())

for i in range(1, 13):
    for j in calendar.monthcalendar(year, i):
        if j[calendar.SATURDAY] != 0:
            dates_list.append(f"/{year}-{i:02d}-{j[calendar.SATURDAY]:02d}/")

for i in range(len(dates_list)):
    csv_file_path = "summary_data/" + "summary_data_" + dates_list[i].strip("/") + ".csv"
    if os.path.exists(csv_file_path):
        data_df_list.append(pandas.read_csv(csv_file_path))





#将data-df-list复制到summary-data-df和summary_data_df_10
summary_data_df = data_df_list[:]
summary_data_df_10 = data_df_list[:]
df_number = random.choice(range(len(data_df_list)))





#赋值匹配的next_week_rank
for i in range(len(summary_data_df) - 1):
    df1 = summary_data_df[i]
    df2 = summary_data_df[i + 1]
    df1["next_week_rank"] = None

    for index, row in df1.iterrows():
        title = row["music_title"]
        producer = row["music_producer"]
        match_row = df2[(df2["music_title"] == title) & (df2["music_producer"] == producer)]
        if not match_row.empty:
            next_week_rank = match_row["this_week_rank"].values[0]
            summary_data_df[i].at[index, "next_week_rank"] = next_week_rank





#部分数据预处理
for i in range(len(summary_data_df)):
    df = summary_data_df[i]
    
    #合并music_title, music_producer
    df = df.assign(music_title_producer = df["music_title"] + "_" + df["music_producer"])
    df = df.set_index("music_title_producer")
    df = df.drop(["music_title", "music_producer"], axis=1)
    
    #将this_week_award的空值设成0
    df = df.fillna({"this_week_award" : 0})
    
    #将last_week_rank值“-”替换“101”
    df["last_week_rank"] = df["last_week_rank"].replace("-", "101")
    
    #将next_week_rank空值替换“101”
    df = df.fillna({"next_week_rank" : 101})
    
    #将weeks_on_chart归一化
    scaler = MinMaxScaler()
    df["weeks_on_chart"] = scaler.fit_transform(df[["weeks_on_chart"]])
    
    #将this_week_rank归一化
    scaler = MinMaxScaler()
    df["this_week_rank"] = scaler.fit_transform(df[["this_week_rank"]])
    
    #将last_week_rank归一化
    scaler = MinMaxScaler()
    df["last_week_rank"] = scaler.fit_transform(df[["last_week_rank"]])
    
    #将best_rank归一化
    scaler = MinMaxScaler()
    df["best_rank"] = scaler.fit_transform(df[["best_rank"]])
    
    #将next_week_rank归一化
    if "next_week_rank" in df.keys():
        scaler = MinMaxScaler()
        df["next_week_rank"] = scaler.fit_transform(df[["next_week_rank"]])
    summary_data_df[i] = df





#变量相关性热力图
merged_df = pandas.concat(summary_data_df[:7], ignore_index=True).drop(["this_week_to_last_week", "this_week_award"], axis=1)
merged_df = merged_df.fillna(0)

#计算变量之间的相关系数矩阵
corr_matrix = merged_df.corr()

#使用热力图可视化相关系数矩阵
seaborn.heatmap(corr_matrix, annot=True, cmap="coolwarm")

plt.title("Variable correlation heatmap\n", fontsize=16, fontweight="bold")
plt.getp(title)
plt.show()





#部分回归图
merged_df = pandas.concat(summary_data_df[:7], ignore_index=True).drop(["this_week_to_last_week", "this_week_award"], axis=1)
merged_df = merged_df.fillna(0)

for column in merged_df.drop("next_week_rank", axis=1).columns:
    
    #自变量x，因变量y
    x = merged_df[column]
    y = merged_df["next_week_rank"]
    
    #添加常数项
    x = sm.add_constant(x)
    
    #构建线性回归模型
    model = sm.OLS(y, x)
    
    #拟合模型
    results = model.fit()
    
    #Figure对象，AxesSubplot对象
    fig, ax = plt.subplots()
    
    #根据拟合值和实际值的距离设置颜色深浅
    distances = abs(results.fittedvalues - y)
    color = cm.Blues(numpy.linspace(0, 1, len(distances)))
    
    #创建散点图和回归线
    ax.scatter(x[column], y, label="actual value", c=color)
    ax.plot(x[column], results.fittedvalues, color="r", label="fit line")
    
    ax.set_xlabel(column)
    ax.set_ylabel("dependent variable")
    ax.set_title(f"\n{column} regression plot\n", fontsize=16, fontweight="bold")
    ax.legend()
    
    plt.show()





#部分数据预处理
for i in range(len(summary_data_df)):
    df = summary_data_df[i]
    
    #对this_week_to_last_week进行独热编码
    encoded_data = pandas.get_dummies(df["this_week_to_last_week"], prefix="this_week_to_last_week")
    df = df.drop("this_week_to_last_week", axis=1)
    df = pandas.concat([df, encoded_data], axis=1)

    #将this_week_award的空值设成0，进行独热编码
    encoded_data = pandas.get_dummies(df["this_week_award"], prefix="this_week_award")
    df = df.drop("this_week_award", axis=1)
    df = pandas.concat([df, encoded_data], axis=1)
    summary_data_df[i] = df
    
    #因为有的week_list的this_week_to_last_week的标签缺失，所以出现空值
    merged_df = pandas.concat(summary_data_df[:7], ignore_index=False)
    merged_df = merged_df.fillna(0)





#线性回归
x = merged_df.drop("next_week_rank", axis=1).values
y = merged_df["next_week_rank"].values

linear_model = LinearRegression()
linear_model.fit(x, y)

x_test = summary_data_df[7].drop("next_week_rank", axis=1).values
y_linear_pred = linear_model.predict(x_test)
y_test = summary_data_df[7]["next_week_rank"].values

print(f"\033[1my_pred:\033[0m {y_linear_pred}")
print(f"\033[1mR^2:\033[0m {r2_score(y_test, y_linear_pred):.2f}")
print(f"\033[1mmse:\033[0m {mean_squared_error(y_test, y_linear_pred)}")
linear_coefficients = linear_model.coef_
print("\033[1mcoefficients:\033[0m \n", linear_coefficients)





#绘制残差图
'''
在拟合线良好的情况下，残差图应该呈现出随机分布的残差，没有明显的模式或趋势。
如果残差图显示出系统性的模式，则说明拟合线可能无法很好地捕捉数据中的关键特征，需要进一步检查模型的假设和进行修正。
残差不应该包含任何可预测的信息。
'''

#预测值与实际值之间的误差
residuals = y_test - y_linear_pred

#绘制残差图
seaborn.residplot(x=y_linear_pred, y=residuals, lowess=True, color="g", line_kws={"color": "r"})

plt.xlabel("predictive value")
plt.ylabel("residual")
plt.title("Residual plot", fontsize=16, fontweight="bold")
plt.getp(title)
plt.show()





#Lasso回归
'''
Lasso回归（Least Absolute Shrinkage and Selection Operator Regression）
Lasso回归使用L1正则化，通过最小化损失函数和系数绝对值的和来实现。
Lasso回归可以促使一些系数变成精确的零，因此可以实现特征选择，即自动减少不重要的特征的系数为零，从而简化模型。
适用于具有高度多重共线性的数据集，在这种情况下，Lasso可以帮助消除不必要的特征。
'''
lasso_model = Lasso(alpha=0.0012)
lasso_model.fit(x, y)
y_lasso_pred = lasso_model.predict(x_test)
print(f"\033[1my_pred: \033[0m {y_lasso_pred}")
print(f"\033[1mR^2:\033[0m {r2_score(y_test, y_lasso_pred):.2f}")
print(f"\033[1mmse:\033[0m {mean_squared_error(y_test, y_lasso_pred)}")
lasso_coefficients = lasso_model.coef_
print(f"\033[1mcoefficients:\033[0m {lasso_coefficients}")





#Ridge回归
'''
Ridge回归（岭回归）:
Ridge回归使用L2正则化，通过最小化损失函数和系数平方的和来实现。
Ridge回归可以缩小系数的大小，但不能将系数压缩到零。它有助于减轻多重共线性问题，提高模型的泛化能力，但不具备特征选择的能力。
适用于数据集中存在多个相关性较强的特征时，可以帮助稳定模型的预测能力。
'''
ridge_model = Ridge(alpha=3.5)
ridge_model.fit(x, y)
y_ridge_pred = ridge_model.predict(x_test)
print(f"\033[1my_pred:\033[0m {y_ridge_pred}")
print(f"\033[1mR^2:\033[0m {r2_score(y_test, y_ridge_pred):.2f}")
print(f"\033[1mmse:\033[0m {mean_squared_error(y_test, y_ridge_pred)}")
ridge_coefficients = ridge_model.coef_
print(ridge_coefficients)





#Linear vs Lasso vs Ridge
x = numpy.arange(0, 1.2, 0.1)

plt.bar(x, linear_coefficients, width=0.02, label="Linear")
plt.bar(x + 0.02, lasso_coefficients, width=0.02, label="Lasso")
plt.bar(x + 0.02 + 0.02, ridge_coefficients, width=0.02, label="Ridge")

plt.title("Linear vs Lasso vs Ridge\n", fontsize=16, fontweight="bold")
plt.xlabel("Feature")
plt.ylabel("Cofficient")
plt.xticks(rotation=45)
plt.xticks(x + 0.02, merged_df.drop("next_week_rank", axis=1).columns.tolist(), ha="right")
plt.legend()
plt.getp(title)
plt.show()





#Linear vs Lasso vs Ridge
x = range(100)

plt.plot(x, y_linear_pred, label="linear-regression")
plt.plot(x, y_lasso_pred, label="random-forest-regression")
plt.plot(x, y_ridge_pred, label="random-forest-regression")
plt.plot(x, y_test, label="true-value", alpha=0.5)
plt.xlabel("Number")
plt.ylabel("Values")
plt.title("\nLinear vs Lasso vs Ridge\n", fontsize=16, fontweight="bold")
plt.legend()
plt.show()





#随机森林回归
'''
数据采样：从训练数据集中随机有放回地抽取一定数量的样本，构成一个新的训练子集。
特征选择：对于每个决策树的节点，在该节点处随机选择一部分特征子集。
决策树构建：使用训练子集和选定的特征子集构建决策树模型。决策树根据选定的特征和相应的阈值进行分割，以最小化预测误差。
预测：对于新的输入样本，通过将其在每个决策树上进行预测，然后取所有决策树预测结果的平均值作为最终的预测结果。
'''
x = merged_df.drop("next_week_rank", axis=1).values
y = merged_df["next_week_rank"].values

x_test = summary_data_df[7].drop("next_week_rank", axis=1).values
y_test = summary_data_df[7]["next_week_rank"].values

random_forest_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
random_forest_model.fit(x, y)
y_random_forest_pred = random_forest_model.predict(x_test)

print(f"\033[1my_pred:\033[0m {y_random_forest_pred}")
print(f"\033[1mR^2:\033[0m {r2_score(y_test, y_random_forest_pred):.2f}")
print(f"\033[1mmse:\033[0m {mean_squared_error(y_test, y_random_forest_pred)}")
print("Number of decision trees:", len(random_forest_model.estimators_))
for i, tree in enumerate(random_forest_model.estimators_):
    print(f"Depth of decision tree {i + 1}: {tree.get_depth()}")





#随机森林回归优化
'''
根据定义的参数空间，创建一个参数网格，包含了所有可能的参数组合。对于上述例子，我们可以创建一个包含所有n_estimators和max_depth组合的网格。
对于每个参数组合，在给定的训练集上训练模型，并使用交叉验证或其他评估指标来评估模型的性能。通常使用的评估指标包括准确率、均方误差、R^2等。
根据评估指标的结果，选择性能最好的参数组合作为最佳参数组合。
'''
x = merged_df.drop("next_week_rank", axis=1).values
y = merged_df["next_week_rank"].values

x_test = summary_data_df[7].drop("next_week_rank", axis=1).values
y_test = summary_data_df[7]["next_week_rank"].values

regressor = RandomForestRegressor(random_state=42)

#参数网格，决策树数量和深度
param_grid = {
    "n_estimators": [10, 20, 30, 40, 50, 100, 150, 200],
    "max_depth": [None, 5, 6, 7, 8, 9]
}

#设置评分标准R^2
scorer = make_scorer(r2_score)

#使用网格搜索寻找最佳参数组合，评分指标：均方差
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring=scorer)
grid_search.fit(x, y)

#最佳参数组合
print("\033[1mBest paramters:\033[0m ", grid_search.best_params_)

#使用最佳参数创建随机森林回归模型
random_forest_model = grid_search.best_estimator_
random_forest_model.fit(x, y)
y_random_forest_pred = random_forest_model.predict(x_test)

print(f"\033[1my_pred:\033[0m {y_random_forest_pred}")
print(f"\033[1mR^2:\033[0m {r2_score(y_test, y_random_forest_pred):.2f}")
print(f"\033[1mmse:\033[0m {mean_squared_error(y_test, y_random_forest_pred)}")

plt.figure(figsize=(12, 6))

#遍历决策树数量
for n_estimators in param_grid["n_estimators"]:
    #R^2
    train_errors = []
    test_errors = []
    #遍历决策树深度
    for max_depth in param_grid["max_depth"]:
        #指定决策树深度，数量
        regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        regressor.fit(x, y)
        
        y_train_pred = regressor.predict(x)
        y_test_pred = regressor.predict(x_test)
        
        #MSE
        train_errors.append(r2_score(y, y_train_pred))
        test_errors.append(r2_score(y_test, y_test_pred))
    plt.plot(param_grid["max_depth"], train_errors, label=f"Train (n_estimators={n_estimators})")
    plt.plot(param_grid["max_depth"], test_errors, label=f"Test (n_estimators={n_estimators})")

plt.title("\nLearning Curve Random Forest", fontsize=16, fontweight="bold")
plt.xlabel("Max Depth")
plt.ylabel("R^2")
plt.legend()
plt.show()





#随机森林回归特征重要性

#获取随机森林回归模型的特征重要性
importances = regressor.feature_importances_
feature_importances = pandas.DataFrame({"feature": merged_df.drop("next_week_rank", axis=1).columns, "importance":importances})
feature_importances = feature_importances.sort_values("importance", ascending=False)

color = list(reversed(cm.Blues(numpy.linspace(0, 1, len(feature_importances)))))

plt.bar(feature_importances["feature"], feature_importances["importance"], color=color)
plt.xticks(rotation=45)
plt.xticks(range(len(feature_importances)), feature_importances["feature"], rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance of Random Forest Regressor\n", fontsize=16, fontweight="bold")
plt.show()





#随机森林回归实际值与预测值对比图
fig, ax = plt.subplots()

#绘制真实值曲线
ax.plot(y_test, label="Actual Values", marker=".", linestyle="none")

#绘制预测值曲线
ax.plot(y_pred, label="Predicted Values", marker=".", linestyle="none")

ax.set_xlabel("Number")
ax.set_ylabel("Values")
ax.set_title("\nActual vs Predicted Values\n", fontsize=16, fontweight="bold")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.show()





#随机森林回归学习曲线

#使用交叉验证计算不同训练集大小下的训练误差和验证误差
'''
k折交叉验证（k-Fold Cross-Validation）：将数据集均匀地划分为k个互斥的子集，每次选择其中一个子集作为测试集，剩余的k-1个子集作为训练集。
重复进行k次训练和测试，每次选择不同的子集作为测试集，最后将k次评估结果的平均值作为最终的性能评估。
'''
train_sizes, train_scores, test_scores = learning_curve(random_forest_model, x, y, cv=5, scoring="neg_mean_squared_error")

#计算每个训练集大小下的平均训练误差
train_mean = -numpy.mean(train_scores, axis=1)

#计算每个训练集大小下的训练误差标准差
train_std = numpy.std(train_scores, axis=1)

#计算每个训练集大小下的平均验证误差
test_mean = -numpy.mean(test_scores, axis=1)

#计算每个训练集大小下的验证误差标准差
test_std = numpy.std(test_scores, axis=1)

#绘制训练误差曲线
plt.plot(train_sizes, train_mean, label="Training error")

#绘制验证误差曲线
plt.plot(train_sizes, test_mean, label="Validation error")

'''
训练误差和验证误差的标准差越大，填充区域就越宽，表示模型在不同训练集大小下的性能可能存在较大的变化。
反之，如果填充区域较窄，则表示模型性能的不确定性较小。
'''
#根据训练误差的标准差填充区域
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)

#根据验证误差的标准差填充区域
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

plt.xlabel("Training set size")
plt.ylabel("Mean Squared Error")
plt.show()





#随机森林回归决策树
tree_number = random.choice(range(100))
tree = random_forest_model.estimators_[tree_number]

#导出决策树的文本表示
tree_text = export_text(tree, feature_names=list(merged_df.drop("next_week_rank", axis=1).columns))
print(tree)

plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=list(merged_df.drop("next_week_rank", axis=1).columns))
plt.title(f"Decision Tree_{tree_number}\n", fontsize=16, fontweight="bold")
plt.getp(title)
plt.show()





#随机森林回归混淆矩阵

#设置阈值
threshold = 50 / 100

#将预测结果转换成二分类问题
binary_predicted = numpy.where(y_random_forest_pred <= threshold, 1, 0)

#计算混淆矩阵的各项指标
tp = numpy.sum((y_test <= threshold) & (binary_predicted == 1))
fn = numpy.sum((y_test <= threshold) & (binary_predicted == 0))
fp = numpy.sum((y_test > threshold) & (binary_predicted == 1))
tn = numpy.sum((y_test > threshold) & (binary_predicted == 0))

#构建混淆矩阵
confusion_matrix = numpy.array([[tp, fn], [fn, tn]])

print(confusion_matrix)

labels = ["Positive", "Negative"]

fig, ax = plt.subplots()

#绘制混淆矩阵的热力图
im = ax.imshow(confusion_matrix, cmap="coolwarm")

#添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)

ax.set(xticks = numpy.arange(confusion_matrix.shape[1]),
      yticks = numpy.arange(confusion_matrix.shape[0]),
      xticklabels = labels,
      yticklabels = labels,
      xlabel = "Predicted label",
      ylabel = "True label")

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(j, i, confusion_matrix[i, j],
               ha="center",
               va="center",
               color="w")

ax.set_title("Confusion Matrix")

plt.show()





#随机森林回归箱线图
residuals = y_test - y_random_forest_pred

#绘制箱线图
plt.boxplot(residuals)

plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Boxplot of Residuals")
plt.show()





#线性回归与随机森林回归对比
x = range(100)

distance_linear = abs(y_test - y_linear_pred)
color_linear = cm.Blues(1 - distance_linear)
distance_random_forest = abs(y_test - y_random_forest_pred)
color_random_forest = cm.Greens(1 - distance_random_forest)

plt.scatter(x, y_linear_pred, color=color_linear, label="linear-regression")
plt.scatter(x, y_random_forest_pred, color=color_random_forest, label="random-forest-regression")
plt.xlabel("Number")
plt.ylabel("Values")
plt.legend()
plt.title("\nLinear vs Random Forest Regression\n", fontsize=16, fontweight="bold")
plt.show()
