import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from imblearn.over_sampling import SMOTE
import utils

features, features_list, labels = utils.load_and_preprocess_data()

isSMOTE = True
if isSMOTE: 
    os = SMOTE(random_state=0)
    os_data_X,os_data_y=os.fit_sample(features, labels)
    features = pd.DataFrame(data=os_data_X)
    labels = pd.DataFrame(data=os_data_y)
    print("length of oversampled data is ",len(os_data_X))
    print(labels[0].value_counts())
    features = np.array(features)
    labels = np.array(labels).ravel()

js_divergences = {}

for i in range(len(features_list)):
    js_divergences[features_list[i]] = jensenshannon(features[:, i], labels)

for feature in sorted(js_divergences, key=js_divergences.get, reverse=True):
    print(feature, js_divergences[feature])

""""
Order of feature importance using Jensen Shannon divergence values without SMOTE:

smoking
no_exercise
exercise
caffeine
cheese_chocolate
sunlight
improper_lighting
ovulation
excess_sleep
exercise.1
overeating
travel
weather_temp
irregular_meals
headache_day
noise
emotional_changes
drinking
massage
odors
medicine
fatigue
less_sleep
other
other.1
menstruation
stress
total_triggers
sleep
rest
sound_sensitivity
light_sensitivity
helping_factors
nausea_vomiting


Using SMOTE:

smoking
no_exercise
improper_lighting
exercise
sunlight
exercise.1
caffeine
cheese_chocolate
ovulation
excess_sleep
travel
overeating
drinking
massage
noise
weather_temp
irregular_meals
emotional_changes
odors
other.1
other
fatigue
menstruation
less_sleep
headache_day
stress
medicine
light_sensitivity
sleep
sound_sensitivity
total_triggers
rest
helping_factors
nausea_vomiting

"""