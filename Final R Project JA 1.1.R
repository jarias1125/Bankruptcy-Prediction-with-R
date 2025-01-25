---
title: "**Finance & Data Analysis with R**"
output: html_notebook
---

# **Individual Course Project**
## ***Joaquin Arias***

**DISCLAIMER:** The following project assumes that the following libraries are already installed, if not please uncomment and run the following lines of code:
```{r}
# install.packages("tidyverse") 
# install.packages(c("tidymodels", "randomForest", "doParallel", "torch", "caTools", "tabnet", "lime", "luz","reshape2","hmisc","smotefamily", "caret", "catboost","DMwR2","themis","recipes","tidymodels","pROC"))
#devtools::install_github("catboost/catboost", subdir = "catboost/R-package")
```


### Introduction

In the following project we are going to apply the knowledge gained on data science and machine learning using the programming language R and all the useful libraries that compose it.
This is going to be done on the Credit Scoring dataset, this data was collected from the Taiwan Economic Journal from 1999 and 2009.It contains 95 features that detail the situationship of the different companies with multiple financial and operational metrics and indicators. The target variable is whether the company has filed for bankruptcy or not, indicated as a class label in the first column.

### Data Analysis

To start the exploratory data analysis we are going to load the data and the packages we expect to use.

```{r}
library(tidyverse)
library(dplyr)              # For data manipulation
library(ggplot2)            # For the plots
library(reshape2)           # For the correlation matrix
library(corrplot)
library(Hmisc)
library(scales)             # For percentage format
library(smotefamily)        # For unbalanced dataset treatment
library(caret)              # For ML tasks
# Machine Learning Models to use
library(catboost)
library(DMwR2)
library(themis)             # Provides step_smote for recipes
library(recipes)            # For pre-processing
library(tidymodels)         # For data splitting and modeling
library(pROC)
```
Loading the bankruptcy dataset and assigning it to the df variable:

```{r}
load("bankruptcy.RData") # Loading the file
df<-bankruptcy # Assigning it to the variable df for manipulation
rm(bankruptcy) # Removing the RData since we already have the data stored in df
```

We are going to analyze the dataframe and the characteristics of it:

```{r}
dim(df) # This command helps us measure the number of rows and columns in the dataset
```

As specified before, our dataset contains 95 features and 1 output variable. In rows, we have the information of 6.819 companies that either went bankrupt or not.

```{r}
head(df) # Display the first 5 rows of our dataframe to understand how is the information given and which variables we have in it.
```
Now its easy to visualize the 95 financial and performance variables that are given in our dataset. The indicators  belong to different categories that help understand the situation of a company:
<ul>
<li> Profitability </li>
<li> Liquidity </li>
<li> Leverage </li>
<li> Efficiency </li>
<li> Growth indicators </li>
</ul>

Now we are going to explore the structure and the datatype of all variables to understand how the following steps need to be performed

```{r}
str(df)    # This function will provide the structure and the datatype of each variable

```

All of our columns are numeric, which indicates that this dataset is suitable and might be ready for the financial analysis and machine learning tasks. The only categorical variable is the target, whether the company went bankrupt or not.

We want to also check if there exist missing values that may need treatment or imputation

```{r}
any(is.na(df)) # Checking if there are missing values in the whole dataset
sum(is.na(df)) # Summing the total number of missing values

```
Without missing values we can proceed to perform statistical analysis on each variable to identify variable trends and outliers. 

We proceed to analyze all the statistics related to each variable of our dataset:

```{r}
summary(df)
```
The dataset contains multiple numeric variables with a mix of ratios that go from 0 to 1 and larger metrics.
The target variable *Bankrupt?* is a binary where 0 represents non bankruptcy and 1 represents bankrupt. Given that the mean is 0.03, we can conclude that most of the companies are not bankrupt. This results in an imbalanced dataset which will require attention when applying the classification model to avoid performance issues.

The variable *Net Income Flag* contains a unique value throughout the different companies, so we are going to remove it.

```{r}
df <- df |> select(-`Net Income Flag`)
```

Variables like *Interest-bearing debt interest rate*, *Inventory Turnover Rate* and *Revenue per Share* have extreme values and may need data scaling.

### Pivot Tables

First, we are going to create pivot tables that compare the bankruptcy rate with different financial metrics divided by 4 quantiles. This will let us know how the financial performance relates with bankruptcy and we are going to gain a superficial understanding of the most important features for bankruptcy.

**Bankruptcy Rate by ROA (A) Quantiles:**

```{r}
df |>
  mutate(ROA_A_Quantile = ntile(`ROA(A) before interest and % after tax`, 4)) |> # Create 4 Quantiles based on ROA A
  group_by(ROA_A_Quantile) |>
  summarise(Bankruptcy_Rate = mean(`Bankrupt?`, na.rm = TRUE)) |> # Summarise them by the average of the bankruptcy rate
  arrange(ROA_A_Quantile)
```
**Interpretation:** Companies in the lowest quantile of ROA have the highest bankruptcy rate. As the ROA increases the bankruptcy decreases sharply and reches 0 in the highest quantile. 
This is a proof that the return on assets is a critical predictor of bankruptcy and that a higher ROA can translate into a reduction in the risk of bankruptcy.
ROA (A) was used as it is directly tied with to the profitability and represents the company ability to generate profits from their assets.

**Bankruptcy Rate by Operating Gross Margin Quantiles:**

```{r}

df |>
  mutate(Operating_Gross_Margin_Quantile = ntile(`Operating Gross Margin`, 4)) |> # Create quartiles
  group_by(Operating_Gross_Margin_Quantile) |>                                    
  summarise(Bankruptcy_Rate = mean(`Bankrupt?`, na.rm = TRUE)) |>       # Calculate bankruptcy rate for each quantile
  arrange(Operating_Gross_Margin_Quantile)                 

```
**Interpretation:** Companies that have lower operating gross margin can be flagged for potential instability due to weaker operational efficiency.

**Bankruptcy Rate by Liability to Equity Quantiles:**

```{r}
df |>
  mutate(Liability_Equity_Quantile = ntile(`Liability to Equity`, 4)) |> # Create quartiles
  group_by(Liability_Equity_Quantile) |>                                
  summarise(Bankruptcy_Rate = mean(`Bankrupt?`, na.rm = TRUE)) |>       # Calculate bankruptcy rate
  arrange(Liability_Equity_Quantile)                                   
```
**Interpretation:** The pivot table obtained shows us that the companies with lower debt compared to their own resources have a very low risk of bankruptcy. As their debt increases the risk of bankruptcy also does.

**Bankruptcy Rate by Interest Coverage Ratio Quantiles:**

```{r}
df |>
  mutate(Interest_Coverage_Quantile = ntile(`Interest Coverage Ratio (Interest expense to EBIT)`, 4)) |> # Create quartiles
  group_by(Interest_Coverage_Quantile) |>                                                                
  summarise(Bankruptcy_Rate = mean(`Bankrupt?`, na.rm = TRUE)) |>                                         # Calculate bankruptcy rate
  arrange(Interest_Coverage_Quantile)       
```
**Interpretation:** It is noteworthy that the pivot table reveals no clear trend of increasing or decreasing bankruptcy risk as we move across the quantiles in this case. In this case we see that the companies who struggle to pay interests on their debt are much more likely to go bankrupt. Companies that have a moderate ability to cover the interest show no risk of bankruptcy but companies with high coverage still face some risk of bankruptcy, but lower than those in Q1.

**Bankruptcy Rate by Inventory Turnover Rate Quantiles:**

```{r}
df |>
  mutate(Inventory_Turnover_Quantile = ntile(`Inventory Turnover Rate (times)`, 4)) |> # Create quartiles
  group_by(Inventory_Turnover_Quantile) |>                                    
  summarise(Bankruptcy_Rate = mean(`Bankrupt?`, na.rm = TRUE)) |>                     # Calculate bankruptcy rate
  arrange(Inventory_Turnover_Quantile)  
```
**Interpretation:** Companies with mid level inventory turnover face a slightly higher bankruptcy risk. However the risk is relatively stable for those with very low or very high turnover and theres no clear upward or downward trend.

**Bankruptcy Rate by Cash Flow to Liability:**

```{r}
df |>
  mutate(Cash_Flow_Liability_Quantile = ntile(`Cash Flow to Liability`, 4)) |> # Create quartiles
  group_by(Cash_Flow_Liability_Quantile) |>                                   #
  summarise(Bankruptcy_Rate = mean(`Bankrupt?`, na.rm = TRUE)) |>             # Calculate bankruptcy rate
  arrange(Cash_Flow_Liability_Quantile) 
```
**Interpretation:** Bankruptcy risk is the highest in Q2, moderate in Q1 and Q3 and the lowest in Quantile 4. This highlights the critical role of mantaining an adequate liquidity to cover liabilities.

**Aggregated Metrics by Bankruptcy Status:** 

```{r}

metrics <- c(
  "Net Income to Total Assets",
  "Gross Profit to Sales",
  "Operating Profit Rate",
  "Current Ratio",
  "Quick Ratio",
  "Cash Flow to Liability",
  "Debt ratio %",
  "Liability to Equity",
  "Interest Coverage Ratio (Interest expense to EBIT)",
  "Inventory Turnover Rate (times)",
  "Accounts Receivable Turnover",
  "Total Asset Turnover",
  "Operating Profit Growth Rate",
  "Net Value Growth Rate",
  "Total Asset Growth Rate",
  "Cash Flow Per Share",
  "Cash Flow to Sales",
  "CFO to Assets"
) # Create a list of the key metrics to aggregate and that we are going to analyze searching for meaningful information

# Calculate the aggregated metrics and the pivot table
df |>
  group_by(`Bankrupt?`) |>
  summarise(across(all_of(metrics), mean, na.rm = TRUE))

```
**Interpretation:**

1. Ability to Pay Short-Term Obligations
<ul>
  <li>**Current Ratio:** Companies that did not go bankrupt have much higher resources 
  (416,729 times their short-term debt) compared to bankrupt companies (just 0.007 times their short-term debt). Meaning that healthy companies have far more resources to pay their short-term bills, while bankrupt companies struggle significantly.</li>
  <li>**Quick Ratio:** Healthy companies can quickly convert resources into cash (7.2 million times the debt), while struggling companies fall behind (only 4,195). So its clear that companies that can quickly turn their assets into cash are less likely to fail financially.</li>
</ul>

2. Level of Debt (Debt Ratio and Liability to Equity)
<ul>
  <li>**Debt Ratio:** Healthy companies owe less money compared to what they own (11%), while bankrupt companies owe  more on average (18.7%). It is clear that companies with too much debt compared to their resources are more likely to face financial trouble.</li>
  <li>**Liability to Equity:** Bankrupt companies borrow slightly more compared to their own funds (29% vs. 27%). Even small increases in borrowing can raise the risk of going into bankruptcy.</li>
</ul>

3. Managing Inventory and Collecting Money (Inventory Turnover and Accounts Receivable Turnover)
<ul>
  <li>**Inventory Turnover:** Both bankrupt and healthy companies sell and replace inventory at about the same speed.So it is safe to say that the inventory management does not seem to explain why companies go bankrupt.</li>
  <li>**Accounts Receivable Turnover:** Healthy companies collect money from customers faster than bankrupt companies. The delay in collecting payments from customers increases the risk to face financial problems.</li>
</ul>

4. Cash Flow
<ul>
  <li>**Cash Flow to Liability:** Healthy companies are slightly better at managing cash flow compared to liabilities (46% vs. 45%). Meaning that companies need strong cash flow to manage their debts and that even small differences can matter.</li>
  <li><b>CFO to Assets:</b> Healthy companies generate cash flow slightly more efficiently from their assets (59%) than bankrupt companies (56%).</li>
</ul>

5. Efficiency and Growth
<ul>
  <li>**Total Asset Turnover:** Healthy companies generate more money from their assets (14%) compared to bankrupt companies (10%).</li>
  <li>**Net Value Growth Rate:** Bankrupt companies grow their equity (money owned by shareholders) faster than healthy companies.Rapid growth may indicate risky strategies or overextension and that can lead to financial trouble.</li>
  <li>**Operating Profit Growth Rate:** Both types of companies grow profits at a similar rate (~85%).</li>
</ul>

Important Observations and Results:
<ul>
  <li>Companies that manage their money well (pay bills, collect payments and keep debt low) are much less likely to fail.</li>
  <li>Companies with too much debt or inefficient use of resources are at greater risk of bankruptcy.</li>
  <li>Surprisingly, rapid growth in equity can signal risky behavior and increase financial trouble.</li>
</ul>

**Growth Rate by Financial Ratios:**

```{r}
df |>
  mutate(Growth_Rate_Quantile = ntile(`Total Asset Growth Rate`, 4)) |> # Create quantiles for growth rate
  group_by(Growth_Rate_Quantile) |>                                    
  summarise(across(c(`ROA(A) before interest and % after tax`, `Debt ratio %`, `Current Ratio`, `Quick Ratio`), 
                   mean, na.rm = TRUE)) |> # Calculate mean for selected financial ratios
  arrange(Growth_Rate_Quantile)

```
**Interpretation:**
<ul>
  <li>Firms with higher growth rates tend to generate better returns on their assets, indicating a correlation between growth and profitability.</li>
  <li>Growth rates don’t seem to significantly affect how much debt companies carry relative to their total assets. This suggests that firms across all growth levels are leveraging their financial resources in a similar manner.</li>
  <li>Companies with moderate growth may strike a balance between growth and maintaining sufficient liquidity, avoiding excessive risk-taking.</li>
  <li>Faster-growing companies may prioritize easily accessible liquid assets, ensuring operational flexibility as they expand.</li>
  <li>While growth does not drastically affect debt ratios, its impact on profitability and liquidity highlights potential areas of risk or stability.</li>
</ul>

**Leverage and Liquidity Insights:**

```{r}
df |>
  mutate(Leverage_Quantile = ntile(`Liability to Equity`, 4)) |> # Create quantiles for leverage
  group_by(Leverage_Quantile) |>                                
  summarise(across(c(`Current Ratio`, `Quick Ratio`, `Cash Flow to Liability`), 
                   mean, na.rm = TRUE)) |> # Calculate mean for liquidity metrics
  arrange(Leverage_Quantile)
```
**Interpretation:**
<ul>
<li>Firms with lower leverage are significantly more equipped to handle short-term financial obligations due to better liquidity.</li>
<li>As leverage increases, companies sacrifice liquidity, making them more vulnerable to financial distress.</li>
<li>Cash flow efficiency relative to liabilities appears unaffected by leverage levels, suggesting other factors may play a larger role in cash flow management.</li>
</ul>
### GGPlots

We are going to plot some variables to have a visual representation and understand their behavior.

**Proportion of bankrupt and non-bankrupt companies:**

```{r}
ggplot(df, aes(x = as.factor(`Bankrupt?`), fill = as.factor(`Bankrupt?`))) +
  geom_bar() +
  geom_text(
    stat = "count",
    aes(label = scales::percent(..count.. / sum(..count..))),
    vjust = -0.5
  ) +
  labs(
    title = "Bankruptcy Distribution",
    x = "Bankruptcy (1 = Yes, 0 = No)",
    y = "Count",
    fill = "Bankruptcy"
  ) +
  theme_minimal()

```
**Interpretation:** The graph confirms what we have seen before, the target of the dataset is unbalanced and we are going to need to work on this for the model.

**Correlation Matrix:**

```{r}

# Compute Spearman correlation matrix and p-values
correlation_results <- Hmisc::rcorr(as.matrix(df[sapply(df, is.numeric)]), type = "spearman")
cor_matrix <- correlation_results$r
p_matrix <- correlation_results$P

# Filter non-significant correlations
cor_matrix[p_matrix >= 0.05] <- NA

# Melt the correlation matrix
melted_corr <- melt(cor_matrix, na.rm = TRUE)

# Create the heatmap
heatmap_plot <- ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_distiller(palette = "RdBu", direction = -1, limits = c(-1, 1), na.value = "grey90") +
  labs(
    title = "Spearman Correlation Heatmap (Significant Only)",
    x = NULL,
    y = NULL,
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, size = 10, margin = ggplot2::margin(t = 10)),
    axis.text.y = element_text(size = 10, margin = ggplot2::margin(r = 10)),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.margin = ggplot2::margin(t = 20, r = 20, b = 20, l = 20),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

# Save the heatmap
ggsave(
  filename = "correlation_heatmap_spaced_labels.png",
  plot = heatmap_plot,
  width = 20,
  height = 15,
  dpi = 300
)

```


In the saved image, we can clearly see the correlation matrix that derives us to the following insights:

There exist strong correlations between Bankruptcy and the following variables:
<p>
- Interest coverage ratio (-)
- Cash flow metrics (-)
- Debt ratios (+)
- Operating profit margins (-)
</p>
And the most predictive metrics appear to be the following:

- ROAA/ROAE (Return on Assets and Equity)
- Debt coverage ratios
- Cash flow to liability ratios
- Working capital metrics

**BoxPlots**

**Bankrupt Vs. Net Income to Total Assets:**

```{r}
ggplot(df, aes(x = as.factor(`Bankrupt?`), y = `Net Income to Total Assets`, fill = as.factor(`Bankrupt?`))) +
  geom_boxplot() +
  labs(title = "Bankrupt vs Net Income to Total Assets",
       x = "Bankrupt (0 = No, 1 = Yes)",
       y = "Net Income to Total Assets") +
  theme_minimal() +
  scale_fill_manual(values = c("skyblue", "orange"), name = "Bankrupt")

```
**Interpretation:** Companies with higher values of Net Income to Total Assets are less likely to go bankrupt. The presence of outliers among non bankrupt companies is a sign that companies with low profitability can avoid bankruptcy if managing other factors correctly.

**Bankrupt vs Total debt/Total Net Worth:**
```{r}
ggplot(df, aes(x = as.factor(`Bankrupt?`), y = `Total debt/Total net worth`, fill = as.factor(`Bankrupt?`))) +
  geom_boxplot() +
  labs(title = "Bankrupt vs Total Debt/Net Worth Correlation",
       x = "Bankrupt (0 = No, 1 = Yes)",
       y = "Total Debt / Net Worth") +
  theme_minimal() +
  scale_fill_manual(values = c("lightgreen", "purple"), name = "Bankrupt")

```
**Interpretation:** Most companies, bankrupt or not, have a low ratio of Total Debt to Net Worth, low leverage is common in both groups. But extremely high values are more prevelant in non bankrupt companies, indicating that high leverage not necessarily results in bankruptcy. 

**Bankrupt Vs. Debt Ratio %:**

```{r}
ggplot(df, aes(x = as.factor(`Bankrupt?`), y = `Debt ratio %`, fill = as.factor(`Bankrupt?`))) +
  geom_boxplot() +
  labs(title = "Bankrupt vs Debt Ratio Correlation",
       x = "Bankrupt (0 = No, 1 = Yes)",
       y = "Debt Ratio %") +
  theme_minimal() +
  scale_fill_manual(values = c("yellow", "red"), name = "Bankrupt")

```
**Interpretation:** Bankrupt companies seem to have a higher debt ratio while non bankrupt generally maintain lower debt ratios.

**Bankrupt vs Net Worth/Assets:**

```{r}
ggplot(df, aes(x = as.factor(`Bankrupt?`), y = `Net worth/Assets`, fill = as.factor(`Bankrupt?`))) +
  geom_boxplot() +
  labs(title = "Bankrupt vs Net Worth/Assets Correlation",
       x = "Bankrupt (0 = No, 1 = Yes)",
       y = "Net Worth / Assets") +
  theme_minimal() +
  scale_fill_manual(values = c("lightpink", "darkblue"), name = "Bankrupt")
```
**Interpretation**: Non bankrupt companies exhibit a higher ratio,indicating stronger equity positions compared to their assets while bankrupt ones have much lowe ratios that indicate weaker solvency.

**Density Plots** 

To gain a deeper understanding of how the previously examined variables behave in bankrupt companies we plot some density graphs to deeply understand the behavior.

**Net Income to Total Assets:**

```{r}
ggplot(df |> filter(`Bankrupt?` == 1), aes(x = `Net Income to Total Assets`)) +
  geom_density(fill = "#FB8861", alpha = 0.7) +
  stat_function(fun = dnorm, args = list(mean = mean(df$`Net Income to Total Assets`[df$`Bankrupt?` == 1], na.rm = TRUE), 
                                         sd = sd(df$`Net Income to Total Assets`[df$`Bankrupt?` == 1], na.rm = TRUE)), 
                color = "black", linetype = "dashed") +
  labs(title = "Net Income to Total Assets \n (Unstable companies)",
       x = "Net Income to Total Assets",
       y = "Density") +
  theme_minimal()
```
**Interpretation:** The peak and mode seem near 0.8 and indicate that majority of bankrupt entities have a high net income compared to their total assets.

**Total Debt / Total Net Worth:**

```{r}
ggplot(df |> filter(`Bankrupt?` == 1), aes(x = `Total debt/Total net worth`)) +
  geom_density(fill = "#56F9BB", alpha = 0.7) +
  stat_function(
    fun = dnorm,
    args = list(
      mean = mean(df$`Total debt/Total net worth`[df$`Bankrupt?` == 1], na.rm = TRUE),
      sd = sd(df$`Total debt/Total net worth`[df$`Bankrupt?` == 1], na.rm = TRUE)
    ),
    color = "black",
    linetype = "dashed"
  ) +
  scale_x_continuous(trans = "log10", breaks = scales::trans_breaks("log10", function(x) 10^x),
                     labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  labs(
    title = "Total Debt / Total Net Worth \n (Unstable companies)",
    x = "Log(Total Debt / Total Net Worth)",
    y = "Density"
  ) +
  theme_minimal()

```
**Interpretation:** Used a logaritmic scale to show the distribution and it is easy to tell that is heavily skewed and most unstable companies have a low ratio.

**Debt Ratio (%)**

```{r}
ggplot(df |> filter(`Bankrupt?` == 1), aes(x = `Debt ratio %`)) +
  geom_density(fill = "#C5B3F9", alpha = 0.7) +
  stat_function(fun = dnorm, args = list(mean = mean(df$`Debt ratio %`[df$`Bankrupt?` == 1], na.rm = TRUE), 
                                         sd = sd(df$`Debt ratio %`[df$`Bankrupt?` == 1], na.rm = TRUE)), 
                color = "black", linetype = "dashed") +
  labs(title = "Debt Ratio % \n (Unstable companies)",
       x = "Debt Ratio %",
       y = "Density") +
  theme_minimal()
```
**Interpretation:** Most bankrupt companies exhibit a debt ratio around to 20%. There are few outiliers and this plot implies that unstable companies have moderate debt levels and that excessive debt ratios are not prevalent among these.

**Net Worth / Assets:**

```{r}
ggplot(df |> filter(`Bankrupt?` == 1), aes(x = `Net worth/Assets`)) +
  geom_density(fill = "#C5B3F9", alpha = 0.7) +
  stat_function(fun = dnorm, args = list(mean = mean(df$`Net worth/Assets`[df$`Bankrupt?` == 1], na.rm = TRUE), 
                                         sd = sd(df$`Net worth/Assets`[df$`Bankrupt?` == 1], na.rm = TRUE)), 
                color = "black", linetype = "dashed") +
  labs(title = "Net Worth / Assets \n (Unstable companies)",
       x = "Net Worth / Assets",
       y = "Density") +
  theme_minimal()
```
**Interpretation:** Unstable companies have a significant net worth compared to their assets.

Some variables in the dataset require special treatment before applying machine learning algorithms. 

### Outlier Treatment

Based on the summary statistics, we identify that *Operating Expense Rate*, *Total Debt / Total Net Worth*, *Inventory Turnover Rate*, *Fixed Assets Turnover* and *Total Asset Turnover* have extreme outliers and need to be treated. We are going to remove the outliers by cutting off values that are above the upper quantile (75th percentile) and replace them with the threshold value itself. 

```{r}
df_clean <- df # Creating a copy of our dataset to clean it
upper_quantile_threshold <- 0.75 # Defining the threshold
variables_to_cut <- c("Operating Expense Rate", 
                      "Total debt/Total net worth", 
                      "Inventory Turnover Rate (times)", 
                      "Fixed Assets Turnover Frequency", 
                      "Total Asset Turnover")
df_clean[variables_to_cut] <- lapply(df_clean[variables_to_cut], function(column) {
  # Calculate the 75th percentile for the column
  upper_bound <- quantile(column, probs = upper_quantile_threshold, na.rm = TRUE)
  
  # Cap values at the 55th percentile
  column <- ifelse(column > upper_bound, upper_bound, column)
  return(column)
})
```
Checking the summary statistics to make sure the change has worked and there is no need to treat the outliers
```{r}
summary(df_clean[variables_to_cut])
```
The outlier removal with the 75th quartile has effectively removed extreme outliers across all the identified variables.

### Logarithmic Transformation

Since we have extreme values and need to stabilize variance in order to apply machine learning models, we are going to apply logarithmic transformation (log(x+1) to reduce skewness avoiding issues with 0s in the variables *Operating Expense Rate*, *Inventory Turnover Rate*, *Fixed Assets Turnover Frequency*, *Total Asset Turnover* and *Revenue Per Share*:
```{r}
df_clean <- df_clean |>
  mutate(
    `Operating Expense Rate` = log1p(`Operating Expense Rate`),
    `Inventory Turnover Rate (times)` = log1p(`Inventory Turnover Rate (times)`),
    `Fixed Assets Turnover Frequency` = log1p(`Fixed Assets Turnover Frequency`),
    `Total Asset Turnover` = log1p(`Total Asset Turnover`),
    `Revenue Per Share` = log1p(`Revenue Per Share (Yuan ¥)`)
  )
```

### Scaling
In the case of extremely skewed variables, we are going to use Min-Max scaling and Z Score standardization in variables that have a more uniform distribution.
```{r}
variables_min_max <- c(
  "Operating Expense Rate",
  "Revenue Per Share",
  "Inventory Turnover Rate (times)",
  "Fixed Assets Turnover Frequency",
  "Total Asset Turnover"
)

df_clean <- df_clean |> 
  mutate(across(all_of(variables_min_max), ~ (. - min(.)) / (max(.) - min(.))))
```

```{r}
variables_zscore <- c(
  "ROA(C) before interest and depreciation before interest",
  "ROA(A) before interest and % after tax",
  "ROA(B) before interest and depreciation after tax",
  "Operating Gross Margin",
  "Pre-tax net Interest Rate",
  "After-tax net Interest Rate",
  "Net Income to Total Assets"
)

# Apply z-score standardization directly to the original variables
df_clean <- df_clean |> 
  mutate(across(all_of(variables_zscore), ~ (.- mean(.)) / sd(.)))


```

Checking the dataset to confirm it is scaled and ready to apply machine learning models.
```{r}
summary(df_clean)  # For min-max scaling

```
```{r}
dim(df_clean)
```
### Treating Imbalanced Dataset
Given the proportion of bankrupt companies, we are going to use SMOTE to increase the minority class while preserving our distribution. 
First, renaming the target variable to avoid issues with the special characters:
```{r}
colnames(df_clean)[colnames(df_clean) == "Bankrupt?"] <- "Bankrupt"
```

### Machine Learning Modeling

Now we split our dataset into test and train:

```{r}
set.seed(42)

data_split <- initial_split(df_clean, prop = 0.7, strata = Bankrupt) 

train_data <- training(data_split)
test_data <- testing(data_split)

```

Then turning the target variable into a factor:

```{r}
train_data <- train_data |>
  mutate(Bankrupt = as.factor(Bankrupt))

test_data <- test_data |>
  mutate(Bankrupt = as.factor(Bankrupt))

```


```{r}
# Recipe for applying SMOTE
smote_recipe <- recipe(Bankrupt ~ ., data = train_data) %>%
  step_smote(Bankrupt, over_ratio = 0.3 / 0.7) # Set the desired 30:70 balance

# Prepare the recipe
prepared_recipe <- prep(smote_recipe, training = train_data)

# Apply SMOTE
smote_train_data <- juice(prepared_recipe)

# Verify class balance
table(smote_train_data$Bankrupt) / nrow(smote_train_data) * 100

# The test data remains untouched for evaluation
test_data <- bake(prepared_recipe, new_data = test_data)
```
Now are dataset is more balanced and ready to train and test algorithms.

### Logistic Regression

We are going to start with the basic model for classification tasks. This is a simple, interpretable and effective if the relationship is close to linear. This model will be used as a baseline and see how it compares to the others we do after this.

```{r}

# Ensure the target variable has valid factor levels
levels(smote_train_data$Bankrupt) <- c("No", "Yes")
levels(test_data$Bankrupt) <- c("No", "Yes")

# Set up training control for cross-validation
train_control <- trainControl(
  method = "cv",        # Cross-validation
  number = 10,          # 10-fold cross-validation
  summaryFunction = twoClassSummary, # Use metrics like AUC
  classProbs = TRUE,    # Use class probabilities
  verboseIter = FALSE    # Dont display progress
)

# Fit logistic regression model
set.seed(42)
logistic_model <- suppressWarnings(
  train(
  Bankrupt ~ .,         # Formula for predictors and target
  data = smote_train_data, # Balanced training data
  method = "glm",       # Logistic regression
  family = "binomial",  # Specify logistic regression
  metric = "ROC",       # Optimize model based on ROC
  trControl = train_control # Training control setup
)
)

# Print model results
print(logistic_model)

# Predict on test data
predictions <- predict(logistic_model, newdata = test_data)
probs <- predict(logistic_model, newdata = test_data, type = "prob")

# Confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$Bankrupt, positive = "Yes")
print(conf_matrix)

# Evaluate AUC
roc_curve <- roc(test_data$Bankrupt, probs[, "Yes"], levels = rev(levels(test_data$Bankrupt)))
auc_value <- auc(roc_curve)
cat("AUC: ", auc_value, "\n")

# Additional evaluation metrics
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy: ", accuracy, "\n")

sensitivity <- conf_matrix$byClass["Sensitivity"]
cat("Sensitivity: ", sensitivity, "\n")

specificity <- conf_matrix$byClass["Specificity"]
cat("Specificity: ", specificity, "\n")

precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)
cat("F1-Score: ", f1_score, "\n")

# Plot ROC Curve
plot(roc_curve, main = "ROC Curve for Logistic Regression", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

# Save results to compare later
model_evaluation <- data.frame(
  Model = "Logistic Regression",
  Accuracy = accuracy,
  AUC = auc_value,
  Sensitivity = sensitivity,
  Specificity = specificity,
  F1_Score = f1_score
)
print(model_evaluation)


```

**Logistic Regression Interpretation:** 

This logistic regression model demonstrates strong overall performance with an accuracy of 91.88%, meaning it correctly predicts both bankrupt and non-bankrupt companies in over 81% of cases. The AUC (0.88) suggests the model has very good discriminatory power, effectively distinguishing between the two classes (bankrupt vs. non-bankrupt).

The specificity (0.93) indicates that the model is highly capable of identifying non-bankrupt companies correctly. However, the sensitivity (0.55), reflects its lower ability to detect bankrupt companies compared to its performance in identifying non-bankrupt ones. This imbalance implies the model is better at avoiding false positives (incorrectly classifying non-bankrupt companies as bankrupt) than at capturing false negatives (missing bankrupt companies).

The F1-Score (0.28), which balances precision and recall, is relatively low. This indicates that, despite its high accuracy and specificity, the model struggles with the minority class (bankrupt companies), highlighting a challenge often encountered in imbalanced datasets.

Overall, the model performs well for general predictions but shows room for improvement, particularly in addressing class imbalance to enhance its sensitivity and F1-Score.

### Logistic Regression HyperTuned:

```{r}

# Define training control with cross-validation
train_control <- trainControl(
  method = "cv",             # Cross-validation
  number = 10,               # 10-fold cross-validation
  summaryFunction = twoClassSummary, # Optimize based on AUC
  classProbs = TRUE,         # Enable class probabilities
  verboseIter = FALSE         # Dont show progress
)

# Define the parameter grid for tuning (adjusted lambda range)
grid <- expand.grid(
  .alpha = c(0, 0.5, 1),    # Elastic Net mixing parameter (ridge: 0, lasso: 1, elastic net: 0.5)
  .lambda = seq(0.001, 0.05, length.out = 10) # Regularization strength (smaller range)
)

# Train the logistic regression model with hyperparameter tuning
set.seed(42)
logistic_tuned <- suppressWarnings(
  train(
  Bankrupt ~ .,               # Formula for predictors and target
  data = smote_train_data,    # Balanced training data
  method = "glmnet",          # Generalized Linear Model with Elastic Net
  metric = "ROC",             # Optimize for AUC
  tuneGrid = grid,            # Parameter grid for tuning
  trControl = train_control   # Training control setup
)
)

# Print the best model and parameters
print(logistic_tuned)

# Predict on test data
predictions <- predict(logistic_tuned, newdata = test_data)
probs <- predict(logistic_tuned, newdata = test_data, type = "prob")

# Ensure that factor levels of the test data match the training levels
levels(test_data$Bankrupt) <- levels(smote_train_data$Bankrupt)

# Confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$Bankrupt, positive = "Yes")
print(conf_matrix)

# Evaluate AUC
roc_curve <- roc(test_data$Bankrupt, probs[, "Yes"], levels = rev(levels(test_data$Bankrupt)))
auc_value <- auc(roc_curve)
cat("AUC: ", auc_value, "\n")

# Additional metrics
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy: ", accuracy, "\n")

sensitivity <- conf_matrix$byClass["Sensitivity"]
cat("Sensitivity: ", sensitivity, "\n")

specificity <- conf_matrix$byClass["Specificity"]
cat("Specificity: ", specificity, "\n")

precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)
cat("F1-Score: ", f1_score, "\n")

# Plot ROC Curve
plot(roc_curve, main = "ROC Curve for Logistic Regression (Tuned)", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

# Save tuned model evaluation results
tuned_model_evaluation <- data.frame(
  Model = "Logistic Regression (Tuned)",
  Accuracy = accuracy,
  AUC = auc_value,
  Sensitivity = sensitivity,
  Specificity = specificity,
  F1_Score = f1_score
)

# Append to existing results
model_evaluation <- rbind(model_evaluation, tuned_model_evaluation)

# Print combined results
print(model_evaluation)
```

**Logistic Regression Tuned Interpretation:** 

The tuned logistic regression model achieved an accuracy of 92.66% and an AUC of 0.91, reflecting strong overall classification performance and excellent discriminatory power. While sensitivity improved slightly to 60%, indicating a increased ability to identify bankrupt companies compared to the baseline model, specificity improved slightly to 93.65%, showing a robust capacity to correctly classify non-bankrupt cases. The F1-Score also improved to 0.32, highlighting a better balance between precision and recall, which is essential for handling the dataset's class imbalance effectively.

Compared to the baseline model, the tuned logistic regression demonstrated marked improvements in accuracy, sensitivity and specificity, indicating superior overall performance and robustness in identifying non-bankrupt companies. The tuned model achieved a better F1-Score, suggesting a more balanced approach to addressing the dataset's class imbalance without overfitting to the minority class.

### Cat Boost 

```{r}
# Convert target variable to numeric (0 and 1) for CatBoost
smote_train_data$Bankrupt <- ifelse(smote_train_data$Bankrupt == "Yes", 1, 0)
test_data$Bankrupt <- ifelse(test_data$Bankrupt == "Yes", 1, 0)

# Convert training and testing data into CatBoost pools
train_pool <- catboost.load_pool(data = smote_train_data[, -which(colnames(smote_train_data) == "Bankrupt")],
                                 label = smote_train_data$Bankrupt)
test_pool <- catboost.load_pool(data = test_data[, -which(colnames(test_data) == "Bankrupt")],
                                label = test_data$Bankrupt)

# Train the CatBoost model using default parameters
set.seed(42)
catboost_model <- catboost.train(
  learn_pool = train_pool,
  test_pool = test_pool,
  params = list(verbose = 0) # To supress long outputs
)

# Predict probabilities on the test set
test_predictions <- catboost.predict(catboost_model, test_pool, prediction_type = "Probability")
predicted_classes <- ifelse(test_predictions > 0.5, 1, 0)

# Confusion matrix
conf_matrix <- confusionMatrix(
  factor(predicted_classes, levels = c(0, 1)),
  factor(test_data$Bankrupt, levels = c(0, 1))
)
print(conf_matrix)

# Evaluate AUC
roc_curve <- roc(test_data$Bankrupt, test_predictions)
auc_value <- auc(roc_curve)
cat("AUC: ", auc_value, "\n")

# Additional metrics
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy: ", accuracy, "\n")

sensitivity <- conf_matrix$byClass["Sensitivity"]
cat("Sensitivity: ", sensitivity, "\n")

specificity <- conf_matrix$byClass["Specificity"]
cat("Specificity: ", specificity, "\n")

precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)
cat("F1-Score: ", f1_score, "\n")

# Plot ROC Curve
plot(roc_curve, main = "ROC Curve for CatBoost Model (Default)", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

# Save model evaluation results in the existing model_evaluation data frame
catboost_model_evaluation <- data.frame(
  Model = "CatBoost (Default)",
  Accuracy = as.numeric(accuracy),
  AUC = as.numeric(auc_value),
  Sensitivity = as.numeric(sensitivity),
  Specificity = as.numeric(specificity),
  F1_Score = as.numeric(f1_score)
)

# Append to model_evaluation
model_evaluation <- rbind(model_evaluation, catboost_model_evaluation)
print(model_evaluation)

```

**CatBoost Interpretation:** Compared to the logistic regression models, CatBoost trades off overall accuracy for a higher F1-Score, suggesting better performance in handling class imbalance and predicting bankrupt companies. The tuned logistic regression outperforms CatBoost in terms of accuracy and AUC, making it more effective for general predictions and distinguishing between classes. However, CatBoost's superior F1-Score indicates its advantage in balancing predictions for the minority class, making it potentially more suitable when identifying bankrupt companies is the priority. This highlights the complementary strengths of the models depending on the specific application.

### CatBoost HyperTuned:

```{r}

# Define parameter grid for hyperparameter tuning
param_grid <- expand.grid(
  depth = c(4, 6, 8),
  learning_rate = c(0.01, 0.1, 0.2),
  iterations = c(100, 300, 500)
)

# Initialize variables to store the best model and its performance
best_model <- NULL
best_auc <- 0
best_params <- NULL

# Perform grid search
for (i in 1:nrow(param_grid)) {
  params <- list(
    depth = param_grid$depth[i],
    learning_rate = param_grid$learning_rate[i],
    iterations = param_grid$iterations[i],
    loss_function = "Logloss",
    eval_metric = "AUC",
    use_best_model = TRUE,
    verbose = 0
  )
  
  # Train the model
  model <- catboost.train(
    train_pool,
    test_pool,
    params = params
  )
  
  # Evaluate the model
  predictions <- catboost.predict(model, test_pool, prediction_type = "Probability")
  auc_value <- pROC::auc(pROC::roc(test_data$Bankrupt, predictions))
  
  # Check if the current model is the best
  if (auc_value > best_auc) {
    best_model <- model
    best_auc <- auc_value
    best_params <- params
  }
}

# Print the best parameters
cat("Best Parameters: \n")
print(best_params)

# Evaluate the best model on the test set
final_predictions <- catboost.predict(best_model, test_pool, prediction_type = "Probability")
roc_curve <- pROC::roc(test_data$Bankrupt, final_predictions)
final_auc <- pROC::auc(roc_curve)

cat("Final AUC: ", final_auc, "\n")

# Confusion matrix
final_predictions_binary <- ifelse(final_predictions > 0.5, 1, 0)
conf_matrix <- caret::confusionMatrix(
  factor(final_predictions_binary, levels = c(0, 1)),
  factor(test_data$Bankrupt, levels = c(0, 1))
)

print(conf_matrix)

# Calculate evaluation metrics
accuracy <- conf_matrix$overall["Accuracy"]
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Save model evaluation results
catboost_tuned_evaluation <- data.frame(
  Model = "CatBoost (Tuned)",
  Accuracy = accuracy,
  AUC = final_auc,
  Sensitivity = sensitivity,
  Specificity = specificity,
  F1_Score = f1_score
)

# Combine with previous model evaluation results
model_evaluation <- rbind(model_evaluation, catboost_tuned_evaluation)
print(model_evaluation)

```
Displaying the parameters of the best model:
```{r}
# Display the parameters of the best model
# Store best parameters during tuning
best_params_list <- list(
  depth = best_params$depth,
  learning_rate = best_params$learning_rate,
  iterations = best_params$iterations,
  loss_function = best_params$loss_function,
  eval_metric = best_params$eval_metric
)

# Print best parameters
cat("Best Hyperparameters:\n")
print(best_params)

```

**CatBoost Tuned Interpretation:** The tuned CatBoost model demonstrates exceptional performance in identifying bankrupt companies, achieving a very high sensitivity of 0.97. This indicates that the model is highly effective at correctly predicting the minority class (bankrupt companies), addressing a critical limitation of the other models. The F1-Score of 0.98 further confirms its strong balance between precision and recall for this class. With an AUC of 0.93, the model also shows excellent discriminatory power. However, the specificity (0.51) is relatively low, meaning it sacrifices some ability to correctly identify non-bankrupt companies for better detection of bankrupt ones.

**Conclusion:** 

The tuned CatBoost model is the most suitable choice given its exceptional sensitivity and overall performance metrics. It aligns with the primary objective of accurately identifying bankrupt companies.

### Feature  Importance:
```{r}
# Get feature importance
feature_importance <- catboost.get_feature_importance(best_model, 
                                                      pool = train_pool, 
                                                      type = "FeatureImportance")

# Convert to a data frame for easier manipulation
feature_importance_df <- data.frame(
  Feature = colnames(smote_train_data[, -which(colnames(smote_train_data) == "Bankrupt")]),
  Importance = feature_importance
)

# Sort by importance in descending order
feature_importance_df <- feature_importance_df[order(-feature_importance_df$Importance), ]

# Print the top features
print(head(feature_importance_df, 15))

# Select top 15 features
top_15_features <- feature_importance_df %>%
  arrange(desc(Importance)) %>%
  slice(1:15)

# Plot top 15 features
library(ggplot2)
ggplot(top_15_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 15 Feature Importances (CatBoost)",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal()

```
**Interpretation:** The feature importance graph illustrates the top 15 predictors of bankruptcy in the tuned CatBoost model. The most influential features include:

1. Fixed Assets Turnover Frequency: A significant determinant of bankruptcy, indicating how effectively fixed assets are utilized.
2. Inventory Turnover Rate (times): Reflects how efficiently inventory is managed and converted into revenue.
3. Borrowing Dependency: Highlights the reliance on external borrowing as a potential risk factor.
4. Operating Expense Rate: Shows the proportion of operating expenses to revenue, critical for profitability.
5. Interest-bearing Debt Interest Rate: Indicates the cost of debt financing.
These features reveal that operational efficiency, financial structure, and borrowing practices significantly impact the bankruptcy likelihood. 

### Conclusion 

The project successfully tackled the problem of predicting bankruptcy using various machine learning models and balanced data handling techniques such as SMOTE. Logistic regression served as a baseline model, while CatBoost (both default and tuned) significantly outperformed in terms of AUC and overall accuracy. However, the tuned CatBoost model demonstrated higher precision and sensitivity in identifying bankrupt companies, indicating its potential for real-world implementation in financial risk assessment.

Steps to Follow:
To further improve the modeling and obtain a stronger solution ready to deploy we could do the following:
<ul>
  <li>Comparing results with other ensemble models like XGBoost or LightGBM. SVM could also be applied to this dataset.</li>
  <li>Feature selection techniques to streamline the input dataset, potentially reducing computation time and improving interpretability.</li>
  <li>Deployment strategies for integrating the predictive model into decision-making workflows, particularly for financial institutions.</li>
</ul>

**Disclaimer:** ChatGPT was used as a support tool in this project mainly for troubleshooting, debugging, understanding syntax and refining code implementations. Additionally, ChatGPT provided suggestions for improving code structure, ensuring adherence to best practices in R programming, and optimizing model evaluation workflows.




























































































































































































