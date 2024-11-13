import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import roman

def prep():

    datadir = "C:\\Users\\radnaa\\Downloads\\"


    # preprocessing
    df_live = pd.read_excel(datadir + '03_livestock.xlsx')
    df_live['sold_p']=df_live['q0605']/df_live['q0604']                 # price of animal sold
    medp = df_live.groupby('ani_id')['sold_p'].median().to_dict()       # assign median price to HH which didn't sell
    df_live['sold_p'].fillna(df_live['ani_id'].map(medp), inplace=True) # 
    df_live['q0602a_y'] = df_live['q0602a'] * df_live['sold_p']         # value of total animal
    df_live['q0603_y'] = df_live['q0603'] * df_live['sold_p']           # value of used animal
    df_live.to_excel(datadir + '03_livestock_adj.xlsx')

    df_by = pd.read_excel(datadir + '05_by_product.xlsx')
    df_by['sold_p']=df_by['q0612']/df_by['q0611']                  # price of prod sold
    medp = df_by.groupby('byprod_id')['sold_p'].median().to_dict()       # assign median price to HH which didn't sell
    df_by['sold_p'].fillna(df_by['byprod_id'].map(medp), inplace=True) # 
    df_by['q0610_y'] = df_by['q0610'] * df_by['sold_p']
    df_by.to_excel(datadir + '05_by_product_adj.xlsx')

    df_crop = pd.read_excel(datadir + '06_crop.xlsx')
    df_crop['sold_p']=df_crop['q0623b']/df_crop['q0623a']
    medp = df_crop.groupby('crop_id')['sold_p'].median().to_dict() 
    df_crop['sold_p'].fillna(df_crop['crop_id'].map(medp), inplace=True) # 
    df_crop['q0620_y'] = df_crop['q0620'] * df_crop['sold_p']
    df_crop.to_excel(datadir + '06_crop_adj.xlsx')


    df_dir = pd.read_excel('mongolia\household\short_id.xlsx')

    df = pd.DataFrame()

    for file in df_dir['file'].unique():
        print(file)
        id_to_name = dict(zip(df_dir[df_dir['file']==file]['var_id'], df_dir[df_dir['file']==file]['var']))
        tmp = pd.read_excel(datadir + file + '.xlsx')
        tmp = tmp[['identif'] + list(df_dir[df_dir['file']==file]['var_id'])]   
        tmp.rename(columns=id_to_name, inplace=True)
        tmp = tmp.groupby(['identif']).sum().reset_index()

        if df.empty: 
            df = tmp
        else:
            df = pd.merge(df, tmp, on='identif', how='left') 
    df.to_csv('income.csv') 

df = pd.read_csv('mongolia\household\income.csv')

df['live_profit'] = df[['live_y_used','live_y_sold']].sum(axis=1) - df['live_c']
df['byprod_profit'] = df[['byprod_y_used','byprod_y_sold','byprod_y_psold']].sum(axis=1)
df['crop_profit'] = df['crop_y_tot'] - df['crop_c']
df['ent_profit'] = df['ent_p'] - df['ent_c']
df = df.drop(columns=['live_y','live_y_used','live_y_sold','live_c','byprod_y_used','byprod_y_sold',
                 'byprod_y_psold','crop_y_tot','crop_y_sold','crop_c','ent_c','ent_p'])

df['w_all'] = df[['w_12m','w_bonus12','w2_12m','w2_bonus12']].sum(axis=1)
df['st_all'] = df[[col for col in df.columns if col.startswith('st')]].sum(axis=1)
df['sc_all'] = df[[col for col in df.columns if col.startswith('sc')]].sum(axis=1)
df['oth_all'] = df[[col for col in df.columns if col.startswith('oth')]].sum(axis=1)
df['profit_all'] = df[[col for col in df.columns if col.startswith('profit')]].sum(axis=1)

df = df[['identif','w_all','st_all','sc_all','oth_all','profit_all','live_profit','byprod_profit','crop_profit','ent_profit']]
df.to_csv('mongolia\household\income_short.csv')


df = pd.read_csv('mongolia\household\income_short.csv')
df.iloc[:,1:] = df.iloc[:,1:]/1e6
df['hh_inc'] = df.iloc[:,1:].sum(axis=1)

# livestock
# 602a - size livestock
# 603,604,605 - used, sold q, sold y
# 606 - cost
# by product
# 610 + 611 + 613 - used, sold, psold q
# 612 + 614 - sold y, psold y
# crop
# 620 total q, 621, 622, 623a - used, for herd, sold q, 623b sold y
# 624 cost 07_agric_exp - 624

var = 'st_all'
medians = []
qts = [0,0.2,0.4,0.6,0.8,1]
qt_vals = df[var].quantile(qts)

plt.figure(figsize=(19, 10))
for i in range(len(qts) - 1):
    lower_quantile = qts[i]
    upper_quantile = qts[i + 1]
    subset = df[(df[var] >= qt_vals[lower_quantile]) & (df[var] <= qt_vals[upper_quantile])]
    med = subset[var].median()
    medians.append(med)
    plt.boxplot(subset[var], positions=[i], widths=0.5, patch_artist=True)
plt.xlabel('Өрхүүдийн бүлэг (орлогоор)')
plt.title('Өрхийн жилийн орлого (2022, сая төгрөг)')
plt.xticks(range(len(qts) - 1), ['Bottom 20%*', '21-40%', '41-60%', '61-80%','Top 20%'])
for i, qt in enumerate(qts):
    plt.text(i+0.3, qt_vals[qt], f"min: {np.round(qt_vals[qt],1)},\nmedian:{np.round(medians[i],1)}",fontsize=10) #, ha='center', va='bottom'
plt.text(-0.5,-20,'*Өнөөх алдартай улны гэх үг чинь шүү дээ :)')
plt.text(3,-30,'Эх сурвалж: Өрхийн нийгэм, эдийн засгийн судалгаа 2022, ҮСХ')
plt.grid(True)
plt.ylim(0, 50)
plt.show()


df_basics = pd.read_stata(datadir + 'basicvars.dta')
df_ind = pd.read_stata(datadir + '02_indiv.dta')

# 0.7% above 1 billion
# 5% above 250 million

list_lim = [50,100]
list_age = ['lead','old']
list_unit = ['саяас','саяас']
list_member = ['тэргүүний','хамгийн өндөр настай гишүүний']
list_lim_value = [50, 100]
i = 0
for lim in range(len(list_lim)):
    for age in range(len(list_age)):
        i += 1
        id_1bn = df[df['hh_inc']>list_lim[lim]].sort_values(by='hh_inc')['identif'].values
        df_bn = df_basics[df_basics['identif'].isin(id_1bn)][['identif','newaimag']]
        df_age_lead = df_ind[(df_ind['identif'].isin(id_1bn)) & (df_ind['ind_id']==1)][['identif','q0104y']] #
        df_age_old = df_ind[(df_ind['identif'].isin(id_1bn))].groupby('identif')[['identif','q0104y']].min().reset_index(drop=True)

        df_bn = df_bn.merge(df_age_lead, on='identif',how='left')
        df_bn = df_bn.merge(df_age_old, on='identif',how='left',suffixes=('_lead','_old'))

        take_age = 'q0104y_' + list_age[age]
        freq = df_bn.groupby(['newaimag', take_age]).size().reset_index(name='frequency')
        prov_tot = freq.groupby('newaimag')['frequency'].sum().reset_index()
        prov_tot.columns = ['newaimag', 'total_frequency']
        tab      = freq.pivot(index='newaimag', columns=take_age, values='frequency').fillna(0)
        sorted_newaimags = tab.sum(axis=1).sort_values(ascending=False).index
        tab      = tab.loc[sorted_newaimags]
        tab.reset_index(inplace=True)
        tab      = tab.merge(prov_tot,on='newaimag',how='left')
        tab['newaimag']  = tab['newaimag'].astype(str) + ' (' + tab['total_frequency'].astype(str) + ')'
        tab.drop(columns ='total_frequency',inplace=True)
        tab.set_index('newaimag',drop=True,inplace=True)

        plt.figure(figsize=(19, 10))
        sns.heatmap(tab, annot=True, cmap='YlGnBu', fmt='g', linewidths=1,annot_kws={"size": 8}) #
        plt.title(f'Зураг {roman.toRoman(i)}. Жилийн {list_lim_value[lim]} {list_unit[lim]} дээш цалингийн орлоготой өрхийн {list_member[age]} төрсөн он ба өрхийн харьяалал (2022, хүн амын 2.5%). Нийт: {tab.sum().sum()}')
        plt.xlabel('Төрсөн он')
        plt.ylabel('Аймаг (нийт тоо)')
        plt.xticks(ticks=[i + 0.5 for i in range(len(tab.columns))], labels=tab.columns)  # Set x-tick positions
        plt.text(-0.5,24.5,'Эх сурвалж: Өрхийн нийгэм, эдийн засгийн судалгаа 2022, ҮСХ')
        plt.savefig(str(list_lim[lim]) + str(list_age[age]) + '.png', dpi=300)

# df_bn[df_bn['newaimag']=='Dundgovi']
