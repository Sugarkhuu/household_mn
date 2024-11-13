import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

datadir = 'data/2023/'

# Load datasets
df_base = pd.read_stata(datadir + r"basicvars.dta")
df = pd.read_stata(datadir + r"01_hhold.dta")

# Rename columns
name_cols = {
    'identif': 'id',
    'hh_no': 'hh_id',
    'q0701': 'house_type',
    'q0702': 'room_num',
    'q0703': 'area',
    'q0704': 'area2',
    'q0705': 'ger_wall_n',
    'q0706': 'if_rent',
    'q0707': 'rent_cost',
    'q0708': 'rent_if_to_cost'
}
df.rename(columns=name_cols, inplace=True)

# Select relevant columns and merge location data
df = df[list(name_cols.values())]
df = df.merge(df_base[['identif', 'location']], left_on='id', right_on='identif', how='left')

# Mapping for categories
house_type_mapping = {
    'Гэр': 'Гэр',
    'Орон сууцны байшин': 'Орон сууц',
    'Бие даасан тохилог сууц': 'Хаус',
    'Сууцны тусдаа байшин': 'Байшин',
    'Нийтийн байр': 'Бусад',
    'Зориулалтын бус сууц': 'Бусад',
    'Бусад': 'Бусад'
}

loc_type_mapping = {
    'Ulaanbaatar': 'Улаанбаатар',
    'Aimagcenter': 'Аймгийн төв',
    'Soumcenter': 'Сумын төв',
    'Countryside': 'Хөдөө'
}

rent_type_mapping = {
    'Тийм': 'Түрээсийн',
    'Үгүй': 'Өөрийн'
}

# Apply mappings
df['house_type'] = df['house_type'].replace(house_type_mapping)
df['location'] = df['location'].replace(loc_type_mapping)
df['if_rent'] = df['if_rent'].replace(rent_type_mapping)

# Order for house types
house_type_order = ['Гэр', 'Байшин', 'Орон сууц', 'Хаус', 'Бусад']
df['house_type'] = pd.Categorical(df['house_type'], categories=house_type_order, ordered=True)

# Step 2: Aggregate data for the first chart
agg_data = df.groupby(['house_type', 'location', 'if_rent']).size().reset_index(name='count')

# Create the first chart (counts)
plt.figure(figsize=(12, 10))
g_count = sns.FacetGrid(agg_data, row='house_type', col='location', margin_titles=True, height=3, aspect=1.2)
g_count.map_dataframe(sns.barplot, x='if_rent', y='count', palette=['#a4bfb2', '#30f093'], width=0.4)

# Add counts at the bottom of each bar for the first chart
for ax in g_count.axes.flatten():
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., 0), 
                    ha='center', va='bottom', fontsize=10, color='black')

g_count.set_axis_labels('', 'Тоо')
g_count.set_titles(row_template='{row_name}', col_template='{col_name}')
g_count.add_legend(title='')

plt.subplots_adjust(top=0.9, bottom=0.2)
g_count.fig.suptitle('Өрхүүдийн сууцны төрөл (өрхийн тоо, эзэмшил ба байршлаар, 8550 өрх, 2023 он)', fontsize=16)
plt.show()

# Step 3: Calculate total counts by location for the second chart
total_counts = agg_data.groupby(['location', 'if_rent'])['count'].sum().unstack(fill_value=0)
total_counts = total_counts.sum(axis=1).reset_index(name='total')
agg_data = agg_data.merge(total_counts, on='location')
agg_data['percentage'] = (agg_data['count'] / agg_data['total']) * 100

# Create the second chart (percentages)
plt.figure(figsize=(12, 10))
g_percentage = sns.FacetGrid(agg_data, row='house_type', col='location', margin_titles=True, height=3, aspect=1.2)
g_percentage.map_dataframe(sns.barplot, x='if_rent', y='percentage', palette=['#a4bfb2', '#30f093'], width=0.4)

# Add percentage labels at the top of each bar for the second chart
for ax in g_percentage.axes.flatten():
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')

g_percentage.set_axis_labels('', '%')
g_percentage.set_titles(row_template='{row_name}', col_template='{col_name}')
g_percentage.add_legend(title='')

plt.subplots_adjust(top=0.9, bottom=0.2)
g_percentage.fig.suptitle('Өрхүүдийн сууцны төрөл (өрхийн хувь (тухайн байршилд), эзэмшил ба байршлаар, 8550 өрх, 2023 он)', fontsize=16)
plt.show()
