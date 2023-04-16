# Basic Libraries
import pandas as pd
import numpy as np
import os
import time
import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

#  Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default='notebook'

from collections import Counter

# Imbalance libraries
from imblearn.over_sampling import SMOTE

# Model selection and metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (accuracy_score, 
                            classification_report,
                            recall_score, precision_score, f1_score,
                            confusion_matrix)

#  Modeling Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

sys.path.insert(0, root_dir)

from config import *

# Show all columns
pd.set_option('display.max_columns', None)

