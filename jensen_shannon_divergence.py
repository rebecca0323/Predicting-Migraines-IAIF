import numpy as np
from scipy.spatial.distance import jensenshannon
import utils

features, features_list, labels = utils.load_and_preprocess_data()

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

"""