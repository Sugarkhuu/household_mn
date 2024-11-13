import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

datadir = "C:\\Users\\radnaa\\Downloads\\"

df_ = pd.read_excel(datadir + "11_savings_loan.xlsx")

codename = {'q0808':'type',
            'q0809':'issue_in_12m',
            'q0810':'amount_12m',
            'q0812':'origin',
            'q0813a':'purpose',
            'q0813b':'purpose2',
            'q0813c':'purpose3',
            'q0814':'pb_1m',
            'q0815':'pb_12m',
            }


df_.rename(columns= codename, inplace=True)
w_norm = 1e6
df_[['pb_1m','pb_12m']] = df_[['pb_1m','pb_12m']].div(w_norm)


df = df_.copy()


df_plot = df[df['pb_1m']>0]
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.boxplot(x='purpose', y='pb_1m', data=df_plot, palette='pastel')
plt.suptitle('Сүүлийн сард төлсөн зээл')
plt.ylim(0, 3)
loan_counts = df['purpose'].value_counts()
for i, purpose in enumerate(loan_counts.index):
    plt.text(i, -1, f'N={loan_counts[purpose]}', ha='center')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust the rect parameter as needed to provide space at the bottom
plt.show()