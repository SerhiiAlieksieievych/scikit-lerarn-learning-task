from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# a
# Завантаження даних Iris
iris_data = load_iris()

# Перетворюємо в pandas DataFrame
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

# Додаємо стовпець з числовими мітками класів (0, 1, 2)
iris_df['target'] = iris_data.target

# Додаємо стовпець з текстовими назвами класів (setosa, versicolor, virginica)
iris_df['species'] = iris_df['target'].apply(lambda x: iris_data.target_names[x])

# Виведення загальної інформації
print("🔹 Форма DataFrame:", iris_df.shape)
print("\n🔹 Типи даних:")
print(iris_df.dtypes)
print("\n🔹 Перші 3 рядки:")
print(iris_df.head(3))

#b
# Виведення ключів об'єкта iris_data
print("🔹 Ключі об'єкта iris_data:")
print(iris_data.keys())

# Виведення опису даних
print("\n🔹 Опис набору даних:")
print(iris_data.DESCR)

#c
# Базова статистика по числових стовпцях
print("🔹 Статистичні показники по кожній ознаці:")
print(iris_df.describe())

#d
# Спостереження для Iris-setosa
setosa_df = iris_df[iris_df['species'] == 'setosa']
print("🔹 Iris-setosa (перші 5 рядків):")
print(setosa_df.head())

# Спостереження для Iris-versicolor
versicolor_df = iris_df[iris_df['species'] == 'versicolor']
print("\n🔹 Iris-versicolor (перші 5 рядків):")
print(versicolor_df.head())

# Спостереження для Iris-virginica
virginica_df = iris_df[iris_df['species'] == 'virginica']
print("\n🔹 Iris-virginica (перші 5 рядків):")
print(virginica_df.head())

#e
# Візуалізація для загального огляду
print("🔹 Побудова pairplot для загального огляду...")
sns.pairplot(iris_df, hue='species', diag_kind='hist')
plt.suptitle("Взаємозв’язки між ознаками Iris", y=1.02)
plt.show()

#f
# Підрахунок кількості прикладів кожного виду
species_counts = iris_df['species'].value_counts()

# Побудова стовпчастої діаграми
print("🔹 Побудова стовпчастої діаграми частоти видів...")
sns.barplot(x=species_counts.index, y=species_counts.values, palette="pastel")

# Декорації графіка
plt.title("Частота кожного виду Iris")
plt.xlabel("Вид")
plt.ylabel("Кількість зразків")
plt.show()

#g
# Ознаки (атрибути): перші 4 колонки
X = iris_df[iris_data.feature_names]

# Мітки: числові класи (0, 1, 2)
y = iris_df['target']

# Виведення розмірностей
print("🔹 Форма X (ознаки):", X.shape)
print("🔹 Форма y (мітки):", y.shape)

# Виведемо перші 5 значень X і y
print("\n🔹 Перші 5 рядків X:")
print(X.head())

print("\n🔹 Перші 5 міток y:")
print(y.head())

#h
from sklearn.model_selection import train_test_split

# Розділення з фіксацією випадковості (random_state для відтворюваності)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Виведення результатів
print("🔹 Кількість рядків у тренувальному наборі:", X_train.shape[0])
print("🔹 Кількість рядків у тестовому наборі:", X_test.shape[0])

print("\n🔹 X_train (перші 5 рядків):")
print(X_train.head())

print("\n🔹 y_train (перші 5 міток):")
print(y_train.head())

#i
# Перевіримо, що всі назви перетворено у числа
print("🔹 Унікальні мітки класів у y:", y.unique())

# Розділення на 80% train / 20% test
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Виведення результатів
print("🔹 Розміри 80/20 розділення:")
print("  - Навчальні X:", X_train_80.shape)
print("  - Тестові X:", X_test_20.shape)

#j
# Розділення 70/30
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Ініціалізація KNN з 5 сусідами
knn = KNeighborsClassifier(n_neighbors=5)

# Навчання моделі
knn.fit(X_train_knn, y_train_knn)

# Прогноз на тестовому наборі
y_pred = knn.predict(X_test_knn)

# Вивід перших 10 прогнозів
print("🔹 Перші 10 передбачень:", y_pred[:10])
print("🔹 Відповідні справжні значення:", y_test_knn.values[:10])

# Оцінка точності
accuracy = accuracy_score(y_test_knn, y_pred)
print(f"\n✅ Точність моделі KNN (k=5): {accuracy:.2%}")