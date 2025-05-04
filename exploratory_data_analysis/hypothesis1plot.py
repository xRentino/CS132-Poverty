import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Color settings
colors = ["#648FFF", "#785EF0", "#A11C5D", "#FE6100", "#FFB000", "#000000", "#FFFFFF"]
colors_grad = sns.color_palette('flare_r', 12)
colors_heat1 = sns.color_palette('flare_r', as_cmap=True)
colors_heat2 = sns.diverging_palette(315, 261, s=74, l=50, center='dark', as_cmap=True)
color_bg = "#1B181C"
color_text = "#FFFFFF"

# Plot settings
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams["figure.figsize"] = (5, 2.5)  # Much smaller figure size
mpl.rcParams['font.family'] = 'Roboto'
mpl.rcParams['figure.titlesize'] = 5
mpl.rcParams['axes.titlesize'] = 5
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 4
mpl.rcParams['xtick.labelsize'] = 3
mpl.rcParams['ytick.labelsize'] = 3
mpl.rcParams['axes.titlepad'] = 3
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 3
mpl.rcParams['ytick.major.pad'] = 3
mpl.rcParams['xtick.major.width'] = 0
mpl.rcParams['xtick.minor.width'] = 0
mpl.rcParams['ytick.major.width'] = 0
mpl.rcParams['ytick.minor.width'] = 0
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = False
mpl.rcParams['legend.title_fontsize'] = 3
mpl.rcParams['legend.fontsize'] = 3
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['lines.linewidth'] = 0.4  # Thinner lines
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
mpl.rcParams["figure.facecolor"] = color_bg
mpl.rcParams["axes.facecolor"] = color_bg
mpl.rcParams["savefig.facecolor"] = color_bg
mpl.rcParams['text.color'] = color_text
mpl.rcParams['axes.labelcolor'] = color_text
mpl.rcParams['xtick.color'] = color_text
mpl.rcParams['ytick.color'] = color_text
mpl.rcParams['axes.edgecolor'] = color_text

# Load Dataset and Data Cleaning
poverty_data = pd.read_csv("exploratory_data_analysis/CS132_DataSheet-graph_data.csv")
poverty_data['Year'] = poverty_data['Year'].ffill() + (poverty_data['Month Only'] / 12)

# Extract data
x = poverty_data['Year']
y_poor = poverty_data['Poor Respondent Count']
y_not_poor = poverty_data['Not Poor Respondent Count']
y_borderline = poverty_data['Borderline Respondent Count']

# Data Split and Reshape
# Poor
X_train_poor, X_test_poor, Y_train_poor, Y_test_poor = train_test_split(x, y_poor, test_size=0.5, random_state=42)
X_train_poor = np.array(X_train_poor).reshape(-1, 1)
X_test_poor = np.array(X_test_poor).reshape(-1, 1)
Y_test_poor = np.array(Y_test_poor)

# Not Poor
X_train_not_poor, X_test_not_poor, Y_train_not_poor, Y_test_not_poor = train_test_split(x, y_not_poor, test_size=0.5, random_state=42)
X_train_not_poor = np.array(X_train_not_poor).reshape(-1, 1)
X_test_not_poor = np.array(X_test_not_poor).reshape(-1, 1)
Y_test_not_poor = np.array(Y_test_not_poor)

# Borderline
X_train_borderline, X_test_borderline, Y_train_borderline, Y_test_borderline = train_test_split(x, y_borderline, test_size=0.5, random_state=42)
X_train_borderline = np.array(X_train_borderline).reshape(-1, 1)
X_test_borderline = np.array(X_test_borderline).reshape(-1, 1)
Y_test_borderline = np.array(Y_test_borderline)

# Regression Models
# Poor
model_poor = linear_model.LinearRegression()
model_poor.fit(X_train_poor, Y_train_poor)
Y_pred_poor = model_poor.predict(X_test_poor)

# Not Poor
model_not_poor = linear_model.LinearRegression()
model_not_poor.fit(X_train_not_poor, Y_train_not_poor)
Y_pred_not_poor = model_not_poor.predict(X_test_not_poor)

# Borderline
model_borderline = linear_model.LinearRegression()
model_borderline.fit(X_train_borderline, Y_train_borderline)
Y_pred_borderline = model_borderline.predict(X_test_borderline)

# Polynomial Regression Models
degree = 3  # Degree of the polynomial

# Poor
poly_model_poor = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
poly_model_poor.fit(X_train_poor, Y_train_poor)
y_pred_poor_poly = poly_model_poor.predict(X_test_poor)

# Not Poor
poly_model_not_poor = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
poly_model_not_poor.fit(X_train_not_poor, Y_train_not_poor)
y_pred_not_poor_poly = poly_model_not_poor.predict(X_test_not_poor)

# Borderline
poly_model_borderline = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
poly_model_borderline.fit(X_train_borderline, Y_train_borderline)
y_pred_borderline_poly = poly_model_borderline.predict(X_test_borderline)

# Print Regression Metrics
print("Poor:")
print(f"Coefficient: {model_poor.coef_[0]:.2f}")
print(f"Intercept: {model_poor.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(Y_test_poor, Y_pred_poor):.2f}")
print(f"R-squared: {r2_score(Y_test_poor, Y_pred_poor):.2f}")
print("\nNot Poor:")
print(f"Coefficient: {model_not_poor.coef_[0]:.2f}")
print(f"Intercept: {model_not_poor.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(Y_test_not_poor, Y_pred_not_poor):.2f}")
print(f"R-squared: {r2_score(Y_test_not_poor, Y_pred_not_poor):.2f}")
print("\nBorderline:")
print(f"Coefficient: {model_borderline.coef_[0]:.2f}")
print(f"Intercept: {model_borderline.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(Y_test_borderline, Y_pred_borderline):.2f}")
print(f"R-squared: {r2_score(Y_test_borderline, Y_pred_borderline):.2f}")

# Create the combined plot
plt.figure()

# Scatter plots with smaller and more transparent circles
scatter_poor = plt.scatter(x, y_poor, color=colors[0], alpha=0.5, s=3)  # Smaller size (s=10) and lower alpha
scatter_not_poor = plt.scatter(x, y_not_poor, color=colors[2], alpha=0.5, s=3)  # Smaller size (s=10) and lower alpha
scatter_borderline = plt.scatter(x, y_borderline, color=colors[4], alpha=0.5, s=3)  # Smaller size (s=10) and lower alpha

# Regression lines
# Generate points for the regression line across the range of years
x_range = np.array([[x.min()], [x.max()]])
y_pred_poor_line = model_poor.predict(x_range)
y_pred_not_poor_line = model_not_poor.predict(x_range)
y_pred_borderline_line = model_borderline.predict(x_range)

# Commented out linear regression lines
# line_poor = plt.plot(x_range, y_pred_poor_line, color=colors[0], linestyle='--')
# line_not_poor = plt.plot(x_range, y_pred_not_poor_line, color=colors[2], linestyle='--')
# line_borderline = plt.plot(x_range, y_pred_borderline_line, color=colors[4], linestyle='--')

# Extend the range of years for prediction by 10 years
x_extended_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

# Predict values for the extended range
y_pred_poor_extended = model_poor.predict(x_extended_range)
y_pred_not_poor_extended = model_not_poor.predict(x_extended_range)
y_pred_borderline_extended = model_borderline.predict(x_extended_range)

# Commented out extended linear regression lines
# line_poor = plt.plot(x_extended_range, y_pred_poor_extended, color=colors[0], linestyle='--')
# line_not_poor = plt.plot(x_extended_range, y_pred_not_poor_extended, color=colors[2], linestyle='--')
# line_borderline = plt.plot(x_extended_range, y_pred_borderline_extended, color=colors[4], linestyle='--')

# Predict values for the extended range using polynomial models
y_pred_poor_extended_poly = poly_model_poor.predict(x_extended_range)
y_pred_not_poor_extended_poly = poly_model_not_poor.predict(x_extended_range)
y_pred_borderline_extended_poly = poly_model_borderline.predict(x_extended_range)

# Plot extended polynomial regression lines
plt.plot(x_extended_range, y_pred_poor_extended_poly, color=colors[0], linestyle='--', label='Poor (Poly)')
plt.plot(x_extended_range, y_pred_not_poor_extended_poly, color=colors[2], linestyle='--', label='Not Poor (Poly)')
plt.plot(x_extended_range, y_pred_borderline_extended_poly, color=colors[4], linestyle='--', label='Borderline (Poly)')

# Events with specified color pattern
# 1986 People Power Revolution
specific_year = 1986
plt.axvline(x=specific_year, color=colors_grad[2], linestyle='--')
plt.annotate(
    f'1986 People Power Revolution',
    xy=(specific_year, 900),
    xytext=(specific_year-1, 900),
    arrowprops=dict(facecolor=colors_grad[2], arrowstyle='->', lw=0.8),
    fontsize=3,
    bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor=colors_grad[2])
)

# 1991 Mount Pinatubo Eruption
specific_year = 1991
plt.axvline(x=specific_year, color=colors_grad[4], linestyle='--')
plt.annotate(
    f'1991 Mount Pinatubo Eruption',
    xy=(specific_year, 800),
    xytext=(specific_year-1, 800),
    arrowprops=dict(facecolor=colors_grad[4], arrowstyle='->', lw=0.8),
    fontsize=3,
    bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor=colors_grad[4])
)

# 1997 Asian Financial Crisis
specific_year = 1997
plt.axvline(x=specific_year, color=colors_grad[5], linestyle='--')
plt.annotate(
    f'1997 Asian Financial Crisis',
    xy=(specific_year, 700),
    xytext=(specific_year-1, 700),
    arrowprops=dict(facecolor=colors_grad[5], arrowstyle='->', lw=0.8),
    fontsize=3,
    bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor=colors_grad[5])
)

# 2008 Global Financial Crisis
specific_year = 2008
plt.axvline(x=specific_year, color=colors_grad[6], linestyle='--')
plt.annotate(
    f'2008 Global Financial Crisis',
    xy=(specific_year, 600),
    xytext=(specific_year-1, 600),
    arrowprops=dict(facecolor=colors_grad[6], arrowstyle='->', lw=0.8),
    fontsize=3,
    bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor=colors_grad[6])
)

# 2020 COVID 19 Pandemic
specific_year = 2020
plt.axvline(x=specific_year, color=colors_grad[7], linestyle='--')  # Keep the line color as is
plt.annotate(
    f'2020 COVID 19 Pandemic',
    xy=(specific_year, 500),
    xytext=(specific_year-1, 500),
    arrowprops=dict(facecolor=colors_grad[7], arrowstyle='->', lw=0.8),
    fontsize=3,
    bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor=colors_grad[7]),
)

# 2023 Post Pandemic Recovery
specific_year = 2023
plt.axvline(x=specific_year, color=colors_grad[8], linestyle='--')
plt.annotate(
    f'2023 Post Pandemic Recovery',
    xy=(specific_year, 400),
    xytext=(specific_year-1, 400),
    arrowprops=dict(facecolor=colors_grad[8], arrowstyle='->', lw=0.8),
    fontsize=3,
    bbox=dict(boxstyle='round,pad=0.2', edgecolor='white', facecolor=colors_grad[8])
)


# Legend for scatter points
legend1 = plt.legend(handles=[scatter_poor, scatter_not_poor, scatter_borderline],
                     labels=['Poor', 'Not Poor', 'Borderline'],
                     title='Data',
                     loc='upper left',
                     bbox_to_anchor=(1.05, 1.0))
plt.gca().add_artist(legend1)

# Define dummy lines for the legend since linear regression lines are commented out
line_poor = plt.Line2D([0], [0], color=colors[0], linestyle='--', label='Poor (Poly)')
line_not_poor = plt.Line2D([0], [0], color=colors[2], linestyle='--', label='Not Poor (Poly)')
line_borderline = plt.Line2D([0], [0], color=colors[4], linestyle='--', label='Borderline (Poly)')

# Legend for trend lines
legend2 = plt.legend(handles=[line_poor, line_not_poor, line_borderline],
                     labels=['Poor', 'Not Poor', 'Borderline'],
                     title='Trends',
                     loc='upper left',
                     bbox_to_anchor=(1.05, 0.7))
plt.gca().add_artist(legend2)

# Customize the plot
plt.xlabel('Year', fontsize=4, labelpad=5, loc='right')  # Position x-axis label to the right
plt.ylabel('Respondent Count', fontsize=4, labelpad=5, loc='top')  # Position y-axis label to the top
plt.title('Projected Self-Rated Poverty Trends in the Philippines over the Years', 
          fontdict={'family': 'Roboto', 'weight': 'bold', 'size': 5}, pad=10, loc='right')

# Adjust the figure margins to move the graph more to the left
plt.subplots_adjust(left=0.15, right=0.25, top=0.85, bottom=0.4)

# Adjust layout to prevent clipping
plt.tight_layout()

plt.show()
# Save the plot
plt.savefig('poverty_combined_graph.png')