import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df_ = pd.read_stata("data/2011/02_indiv.dta")

codename = {'q0102':'hh_role',
            'q0103':'gender',
            'q0105y':'age',
            'q0204':'educ',
            'q0416a':'w_1m_last',
            'q0416b':'w_12m',
            'q0416c':'w_bonus12'
            }

df_.rename(columns= codename, inplace=True)

df = df_[list(codename.values())]
myvar = 'w_1m_last'

# df['w_1m_avg'] = df['w_12m']/df['h_month']

w_norm = 1e6
# df[['w_1m_last','w_1m_avg','w_12m']] = df[['w_1m_last','w_1m_avg','w_12m']].div(w_norm)
df[['w_1m_last','w_12m']] = df[['w_1m_last','w_12m']].div(w_norm)

ed_dict = {
    'БОЛОВСРОЛГҮЙ': 'Бгүй',
    'ДИМЛОМЫН ДЭЭД': 'БАКАЛАВР',
    'МЭРГЭЖЛИЙН БОЛОВСРОЛ': 'БД',
    'БҮРЭН ДУНД': 'БД',
    'СУУРЬ': 'БАГА'
}

# Replace values in the 'educ' column
df['educ'] = df['educ'].replace(ed_dict)

age_bins = [0, 30, 40, 50, float('inf')]
age_labels = ['Up to 30 y.o', '31-40 y.o', '41-50 y.o', '51+ y.o']

# Add Age Group column to the DataFrame
df['AGE GROUP'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

df_plot = df #[df['work']=='Тийм']
median_wages = df.groupby('educ')[myvar].median().sort_values(ascending=False)
col_order = ['ЭМ','ЭР']
row_order = ['Up to 30 y.o', '31-40 y.o', '41-50 y.o', '51+ y.o']
g = sns.FacetGrid(df_plot, row='AGE GROUP', col='gender',row_order=row_order, col_order=col_order, margin_titles=True)
g.map_dataframe(sns.boxplot, x='educ', y=myvar, order=median_wages.index, linewidth=1)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels('', 'Wage (сая төгрөгөөр)')
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=15)
    # Set y-axis ticks with step size of 0.5
    # ax.set_yticks(np.arange(0, 7, 1))
    # plt.set_yticks(np.arange(0, 5, 1))
    ax.axhline(y=0.2, color='black', linestyle='--')
plt.yticks(np.arange(0, 7, 0.5))
g.set(ylim=(0, 1.5))
# plt.suptitle('2022 онд олсон сарын дундаж цалин (5 саяас дээш цалинтай нь зурагт харагдахгүй)')
plt.suptitle('2011 онд олсон сарын дундаж цалин')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter as needed to provide space at the bottom
# plt.show()
plt.savefig('figures/2011_salary.png')

df_plot = df[(df['work']=='Тийм') & (df['w_1m_avg']>=2)] 
# df_plot = df[df['work']=='Тийм'] 
median_wages = df.groupby('educ')['w_1m_avg'].median().sort_values(ascending=False)
col_order = ['Эмэгтэй','Эрэгтэй']
row_order = ['Up to 30 y.o', '31-40 y.o', '41-50 y.o', '51+ y.o']
g = sns.FacetGrid(df_plot, row='AGE GROUP', col='gender',row_order=row_order, col_order=col_order, margin_titles=True)
g.map_dataframe(sns.countplot, x='educ',order=median_wages.index, linewidth=1)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels('', 'Wage (сая төгрөгөөр)')
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=15)
    ax.axhline(y=50, color='black', linestyle='--')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter as needed to provide space at the bottom
plt.suptitle('Цалинтай ажил эрхэлж буй хүний тоо (2 саяас дээш цалинтай хүмүүс, хамрагдсан өрх 2.4%)')
# plt.suptitle('Цалинтай ажил эрхэлж буй хүний тоо (хамрагдсан өрх 2.4%)')
plt.show()