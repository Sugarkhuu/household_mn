import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


datadir = "C:\\Users\\sugarkhuu\\Downloads"

df_ = pd.read_excel(datadir + "\\02_indiv.xlsx")

codename = {'q0102':'hh_role',
            'q0103':'gender',
            'q0105y':'age',
            'q0210':'educ',
            'q0102':'hh_member',
            }

df_.rename(columns= codename, inplace=True)

df = df_[list(codename.values())]
df['id'] = df_['identif']

fam_dict = {
    'Өрхийн тэргүүн': 'head',
    'Эхнэр/ нөхөр': 'spouse'}
df['hh_member'] = df['hh_member'].replace(fam_dict)

ed_dict = {
    'Боловсролгүй': 'Суурь ба бага',
    'Бага': 'Суурь ба бага',
    'Суурь': 'Суурь ба бага',
    'Тусгай мэргэжлийн дунд': 'Бүрэн дунд',
    'Техникийн болон мэргэжлийн': 'Бүрэн дунд',
    'Дипломын дээд': 'Бакалавр'}

# Replace values in the 'educ' column
df['educ'] = df['educ'].replace(ed_dict)

df_head = df[df['hh_member'] == 'head']
df_spouse = df[df['hh_member'] == 'spouse']

dfw = df_head.merge(df_spouse,on='id',how='left',suffixes={'_h','_s'})
dfw['educ_s'].fillna('ганц бие', inplace=True)
dfw['age_s'].fillna(0, inplace=True)

dfw['educ_male'] = dfw.apply(lambda row: row['educ_h'] if row['gender_h'] == 'Эрэгтэй' else row['educ_s'], axis=1)
dfw['educ_female'] = dfw.apply(lambda row: row['educ_h'] if row['gender_h'] == 'Эмэгтэй' else row['educ_s'], axis=1)
dfw['age_male'] = dfw.apply(lambda row: row['age_h'] if row['gender_h'] == 'Эрэгтэй' else row['age_s'], axis=1)
dfw['age_female'] = dfw.apply(lambda row: row['age_h'] if row['gender_h'] == 'Эмэгтэй' else row['age_s'], axis=1)

df_plot = dfw[dfw['age_female'] - dfw['age_male'] > 10]  #dfw #[dfw['age_h']<30] #dfw[dfw['age_male']-dfw['age_female']>=10]
education_order = ['Доктор','Магистр', 'Бакалавр','Бүрэн дунд','Суурь ба бага','ганц бие']

conf_matrix = confusion_matrix(df_plot['educ_male'], df_plot['educ_female'], labels=education_order)
total_count = conf_matrix.sum()
conf_matrix_percent_total = (conf_matrix / total_count) * 100
conf_plot = conf_matrix
heatmap = sns.heatmap(conf_plot.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=education_order, yticklabels=education_order) #  
row_totals = conf_plot.sum(axis=1)
col_totals = conf_plot.sum(axis=0)
for i, total in enumerate(row_totals):
    heatmap.text(i + 0.5, len(education_order) + 0.5, str(np.round(total,2)), ha='center', va='center')
for i, total in enumerate(col_totals):
    heatmap.text(len(education_order) + 0.5, i + 0.5, str(np.round(total,2)), ha='center', va='center')
heatmap.text(len(education_order) + 0.5, len(education_order) + 0.5, f'Total: {total_count}', ha='center', va='center')

plt.xticks(rotation=0) 
plt.xlabel('Эрэгтэй')
plt.ylabel('Эмэгтэй')
plt.title('Өрхийн гишүүдийн боловсролын байдал (эрэгтэй нь 10+ насаар ах, 2022 он)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Convert to DataFrame for better visualization
confusion_df = pd.DataFrame(conf_matrix, index=education_order, columns=education_order)

# Add total row and column
confusion_df.loc['Total'] = confusion_df.sum()
confusion_df['Total'] = confusion_df.sum(axis=1)

print(confusion_df)



edu_mat = dfw.groupby(['educ_male', 'educ_female']).size().unstack(fill_value=0)

# Add missing education levels
all_education_levels = set(df['educ_male'].unique()) | set(df['educ_female'].unique())
missing_education_levels = all_education_levels - set(edu_mat.index)
for edu in missing_education_levels:
    edu_mat.loc[edu] = 0
missing_education_levels = all_education_levels - set(edu_mat.columns)
for edu in missing_education_levels:
    edu_mat[edu] = 0

education_order = ['Доктор','Магистр', 'Бакалавр', 'Дипл дээд','ТМ дунд','ТМ', 'Бүрэн дунд','Суурь', 'Бага','Бгүй','байхгүй']

edu_mat = edu_mat.reindex(education_order, axis=0)
edu_mat = edu_mat.reindex(education_order, axis=1)
print(edu_mat)
