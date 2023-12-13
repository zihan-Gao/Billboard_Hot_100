#导入库
import requests
from bs4 import BeautifulSoup
import pandas
import calendar
import time
import random
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import r2_score
import numpy
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer

#获得网页数据
