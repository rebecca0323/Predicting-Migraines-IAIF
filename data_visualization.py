import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('total_diary_migraine.csv', header=0)
features = pd.DataFrame(data)
features['no_headache_day'].fillna('N', inplace=True)

features['migraine'].fillna(0, inplace=True)
features['headache_day'] = features['headache_day'].map({'Y':0, 'N':1})

labels = np.array(features['migraine'])

"""
# plot for migraines vs no migraine

print(features['migraine'].value_counts())

fig, ax = plt.subplots()
sns.countplot(x='migraine', data=features, palette='hls')
ax.set(xlabel="Migraine Occurrence", ylabel = "Number of Occurrences")
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = 'No'
labels[1] = "Yes"
ax.set_xticklabels(labels)
plt.savefig('count_plot_migraines')

# plot for total triggers vs migraine
pd.crosstab(features.total_triggers, features.migraine).plot(kind='bar')
plt.title('Migraine Frequency for Number of Triggers')
plt.xlabel('Number of Triggers')
plt.ylabel('Frequency of Migraine')
plt.xlim(-0.5,5.5)
plt.savefig('migraine_fre_tot_trig')
"""

# stacked bar chart for total triggers vs migraine
table = pd.crosstab(features.total_triggers,features.migraine)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Number of Triggers vs Migraine Occurance')
plt.xlabel('Total Number of Triggers')
plt.ylabel('Proportion of Data')
plt.legend(['No Migraine','Migraine Occurrence'])
plt.savefig('total_trigger_vs_pur_stack')
plt.show()