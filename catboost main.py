from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    'Имя': ['Фариз', 'Данияр', 'Досымжан', 'Еркеназ'],
    'Кредитная_история': [1, 0, 1, 1],
    'Возраст': [30, 25, 35, 28],
    'Доход': [50000, 30000, 45000, 40000],
    'Берет_кредит': [1, 0, 1, 0]  
})

X = data[['Кредитная_история', 'Возраст', 'Доход']]
y = data['Берет_кредит']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')
