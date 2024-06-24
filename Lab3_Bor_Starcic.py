# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import random as rnd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import ks_2samp
from scipy.stats import anderson
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency


if __name__ == "__main__":
    X_train = np.load("učni_vzorci.npy")
    y_train = np.load("učne_oznake.npy")
    X_test = np.load("testni_vzorci.npy")
    y_test = np.load("testne_oznake.npy")

    # Združimo učno in testno zbirko
    X_combined = np.concatenate((X_train, X_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)


# %%
### Razdelitev lokacij slik v listu v posamezne skupine za lažjo obravnavo naprej ###
razredNo = [[], [], [], [], [], [], [], [], [], []]

for i in range(len(y_combined)):
    razredNo[y_combined[i]].append(i)


# %%
### Računanje značilk posamezne slike  ###
def calculate_image_features(class_number, sample_id):

    histogram = []  
    num_pixels_non_black = 0  

    for y in range(28):
        for x in range(28):
            pixel_value = X_combined[razredNo[class_number][sample_id], y, x]

            if pixel_value:  # Če pixel ni črn
                histogram.append(pixel_value)
                num_pixels_non_black += 1

    if num_pixels_non_black == 0:
        return [0, 0, 0, 0, 0]  # Vrne nule, če ni nobenega nečrnega piksla

    # Normalizira vrednosti v histogramu 
    normalized_histogram = [float(count) / num_pixels_non_black for count in histogram]
    
    # Izračuna značilke histograma
    mean_brightness = np.mean(normalized_histogram)
    variance = np.var(normalized_histogram)
    skewness = stat.skew(normalized_histogram)
    kurtosis = stat.kurtosis(normalized_histogram)

    return num_pixels_non_black, mean_brightness, variance, skewness, kurtosis




# %%
### Izračun značilk za vse vzorce ###
num_variables = 5

variables = np.zeros((10 * 7000, num_variables))

for i in range(10):  
    for j in range(7000):
        sample_index = razredNo[i][j]
        variables[sample_index] = calculate_image_features(i, j)
        

# %%
def CalculateNeighborFeatures(class_num, sample_id):
    directions = [[] for _ in range(8)]  # 8 smeri
    second_order_features = np.zeros(32)  # 32 značilk (4 na smer)

    for row in range(28):
        for col in range(28):
            # Trenutna vrednost piksla
            pixel_value = X_combined[razredNo[class_num][sample_id], row, col]

            # Izrčunaj sosednje vrednosti, če trenutna vrednost ni črna
            if pixel_value != 0:
                for direction in range(8):
                    yOffset, xOffset = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)][direction]
                    newY, newX = row + yOffset, col + xOffset
                    if 0 <= newY < 28 and 0 <= newX < 28:
                        neighbor_value = X_combined[razredNo[class_num][sample_id], newY, newX]
                        delta = int(neighbor_value) - int(pixel_value)
                        directions[direction].append(delta)

    # Izračunaj znčilke za vsako smer
    for i in range(8):
        second_order_features[i * 4] = np.mean(directions[i]) if directions[i] else 0  # Mean
        second_order_features[i * 4 + 1] = np.var(directions[i]) if directions[i] else 0  # Variance
        second_order_features[i * 4 + 2] = stat.skew(directions[i]) if directions[i] else 0  # Skewness
        second_order_features[i * 4 + 3] = stat.kurtosis(directions[i]) if directions[i] else 0  # Kurtosis

    return second_order_features


# %%
### Inicializacija polja za shranjevanje lastnosti drugega reda za učne podatke ###
features_second_order = np.zeros((10 * 7000, 32)) 

for i in range(10):
    for j in range(7000):
        location = razredNo[i][j]
        feature_vector = CalculateNeighborFeatures(i, j)
        features_second_order[location] = feature_vector


# %%
### Združevanje značilk ###

total_features_count = 37  

# Inicializacija polja za združene značilke
combined_features = np.zeros((X_combined.shape[0], total_features_count))
combined_features = np.concatenate((variables, features_second_order), axis=1)

# Shranjevanje polja združenih značilk
np.save("combined_features.npy", combined_features)

# %%
### Brisanje značilk za testiranje različnih kombinacij ###

combined_features = np.load("combined_features.npy")

# Ustvari seznam indeksov za vseh 37 značilk
all_features = list(range(37))

# Značilke izbrane pri 2. lab: [4, 8, 12, 20, 21, 28, 29, 36]
kept_features = [4, 8, 12, 20, 21, 28, 29, 36]

# Izračunaj indekse značilk, ki jih je treba izbrisati
deleted_features = [index for index in all_features if index not in kept_features]

# Izbris določenih značilk iz naborov podatkov
combined_features_reduced = np.delete(combined_features, deleted_features, axis=1)

# %%
### Klasifikator najbližjih sosedov z značilkami drugega reda - 500 vzorcev na razred ###

# Parametri
num_features = 37 - len(deleted_features) 
num_samples_per_class = 500  
num_classes = 10
K = 5  # Število sosedov

# Pripravi manjši list podatkov 
y_small = np.zeros((num_classes * num_samples_per_class,))
X_features_small = np.zeros((num_classes * num_samples_per_class, num_features))

# Izberi 500 naključnih vzorcev iz vsakega razreda
for i in range(num_classes):
    class_indices = [index for index, label in enumerate(y_combined) if label == i]
    if len(class_indices) < num_samples_per_class:
        raise ValueError(f"Ni dovolj vzorcev za razred {i}.")
    random_sample_indices = rnd.sample(class_indices, num_samples_per_class)
    for j, index in enumerate(random_sample_indices):
        y_small[i * num_samples_per_class + j] = y_combined[index]
        X_features_small[i * num_samples_per_class + j] = combined_features_reduced[index]

# Križno preverjanje
kf = KFold(n_splits=14, shuffle=True, random_state=None)
accuracies = []

for train_index, test_index in kf.split(X_features_small):
    # Razdeli podatke na učne in testne
    X_train_cv, X_test_cv = X_features_small[train_index], X_features_small[test_index]
    y_train_cv, y_test_cv = y_small[train_index], y_small[test_index]

    # Standardizacija in normalizacija
    scaler = StandardScaler()
    normalizer = MinMaxScaler()  # or use Normalizer()

    X_train_cv = scaler.fit_transform(X_train_cv)
    X_train_cv = normalizer.fit_transform(X_train_cv)

    X_test_cv = scaler.transform(X_test_cv)
    X_test_cv = normalizer.transform(X_test_cv)

    # Usposobi klasifikator
    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(X_train_cv, y_train_cv)

    # Preizkusi klasifikator
    y_test_predicted = classifier.predict(X_test_cv)
    accuracy = np.sum(y_test_predicted == y_test_cv) / len(y_test_cv)
    accuracies.append(accuracy)

# Prikaz vseh 14 ocen natančnosti
for idx, acc in enumerate(accuracies, 1):
    print(f"Natančnost klasifikacije {idx}: {acc * 100:.2f}%")

# Izračunaj povprečno natančnost
average_accuracy = np.mean(accuracies)
print(f"Povprečna natančnost klasifikacije: {average_accuracy * 100:.2f}%")

# Statistična analiza natančnosti
std_dev_accuracy = np.std(accuracies)
median_accuracy = np.median(accuracies)

print(f"Standardni odklon natančnosti: {std_dev_accuracy * 100:.2f}%")
print(f"Mediana natančnosti: {median_accuracy * 100:.2f}%")

# Graf natančnosti
plt.figure(figsize=(10, 6))
plt.boxplot(accuracies, vert=False, patch_artist=True)
plt.title('Natančnosti klasifikacije')
plt.xlabel('Natančnost')
plt.show()

# %%
### Test Kolmogorova in Smirnova za primer dveh vzorcev ###
# Razdelimo natančnosti na dva vzorca
sample1 = accuracies[:12]  # Učne zbirke
sample2 = accuracies[12:]  # Testni zbirki 

# Izvedemo test Kolmogorova-Smirnova
ks_statistic, p_value = ks_2samp(sample1, sample2)

print(f"KS statistika: {ks_statistic}")
print(f"P-vrednost: {p_value}")

# Interpretacija rezultatov
alpha = 0.05
if p_value < alpha:
    print("Vzorca se statistično značilno razlikujeta.")
else:
    print("Ni dovolj dokazov, da bi trdili, da se vzorca statistično značilno razlikujeta.")

# %%
### Test Andersona in Darlinga za primer 𝑘 vzorcev ###
# Primer enega vzorca
sample = accuracies  

# Izvedemo test Andersona-Darlinga
result = anderson(sample, dist='norm')

print('Statistika: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('Pri stopnji značilnosti %.1f%% vzorec sledi normalni porazdelitvi.' % sl)
    else:
        print('Pri stopnji značilnosti %.1f%% vzorec ne sledi normalni porazdelitvi.' % sl)

# %%
### Test vsote rangov (Mann-Whitney 𝑈-test) ###

# Izvedemo Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(sample1, sample2)

print(f"Mann-Whitney U statistika: {u_statistic}")
print(f"P-vrednost: {p_value}")

# Interpretacija rezultatov
alpha = 0.05
if p_value < alpha:
    print("Obstaja statistično značilna razlika med skupinama.")
else:
    print("Ni statistično značilne razlike med skupinama.")



# %%
