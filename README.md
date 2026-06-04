![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-yellow)

# Анализ оттока клиентов банка

---

### Описание проекта

Проект предсказывает вероятность того, что клиент уйдёт из банка (целевой признак `Exited`). Основная цель — получить рабочую модель удержания клиентов и интерпретировать результаты для бизнес-решений.

Анализ выполнен для **немецкого филиала** банка (`Geography == 'Germany'`).

---

### Цели и задачи проекта

1. **Предобработка данных:**
   - Отобрать клиентов немецкого филиала банка.
   - Исключить неинформативные признаки.
   - Создать новые осмысленные признаки для моделирования.
   - Для EDA — категориальный признак `CreditScoreCat` (в модель не входит, используется только для визуализации).
   - Разделить выборку на обучающую и тестовую с учётом стратификации.

2. **Разведочный анализ данных (EDA):**
   - Построить сводные таблицы и тепловые карты зависимости оттока от `CreditScore` и `Tenure`.
   - Определить группы клиентов с наибольшим риском ухода.

3. **Моделирование и оценка:**
   - Построить базовую модель **логистической регрессии**.
   - Добавить **полиномиальные признаки** и оценить влияние на качество.
   - Реализовать **дерево решений** и **случайный лес** с различными параметрами глубины и числа деревьев.
   - Выполнить **подбор оптимального порога вероятности** для максимизации метрики F1-score.
   - Сравнить модели по качеству и устойчивости к переобучению.

4. **Прогнозирование:**
   - Использовать **обрезанное дерево решений** для прогноза вероятности оттока конкретного клиента (на примере клиента Василия).

5. **Интерпретация и выводы:**
   - Определить ключевые признаки, влияющие на отток.
   - Сформулировать рекомендации для бизнеса по удержанию клиентов.

**Итог:**  
Проект демонстрирует полный цикл разработки модели машинного обучения — от анализа данных и инженерии признаков до выбора оптимальной модели и интерпретации результатов.

---

### Структура проекта

```
bank-customer-churn-analysis/
├── data/
│   └── churn.csv                          # исходный датасет
├── notebooks/
│   └── bank_churn_analysis.ipynb          # основной Jupyter Notebook
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

### Источник данных

Датасет [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/gauravduttakiit/churn-modelling) (Churn Modelling).  
Файл в репозитории: `data/churn.csv`.

---

### Описание данных

Исходные признаки в наборе:

| Признак | Описание |
|---------|----------|
| `RowNumber` | порядковый номер строки (удаляется при моделировании) |
| `CustomerId` | идентификатор клиента (удаляется) |
| `Surname` | фамилия (удаляется) |
| `CreditScore` | кредитный рейтинг |
| `Geography` | страна (`France`, `Spain`, `Germany`; для модели — только `Germany`) |
| `Gender` | пол |
| `Age` | возраст |
| `Tenure` | сколько лет клиент в банке |
| `Balance` | баланс на счёте |
| `NumOfProducts` | число продуктов банка у клиента |
| `HasCrCard` | наличие кредитной карты (0/1) |
| `IsActiveMember` | активность клиента (0/1) |
| `EstimatedSalary` | оценочный доход |
| `Exited` | цель (1 — ушёл, 0 — остался) |

Признаки, созданные для моделирования:

- `BalanceSalaryRatio = Balance / EstimatedSalary`
- `TenureByAge = Tenure / Age`
- `CreditScoreGivenAge = CreditScore / Age`
- `Gender_Male` — бинарное кодирование пола (`pd.get_dummies`, `drop_first=True`)

Признак для EDA (не входит в модель):

- `CreditScoreCat` — категории кредитного рейтинга (`Very_Poor`, `Poor`, `Fair`, `Good`, `Excellent`, `Top`, `Deep` для значений < 300)

---

### Используемые технологии

- **Python 3.10+**
- **Pandas**, **NumPy** — обработка данных
- **Matplotlib**, **Seaborn**, **Plotly** — визуализация
- **Scikit-learn** — модели и метрики
- **Jupyter Notebook** — интерактивный отчёт

---

### Как запустить проект

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/theKerimKerimov/bank-customer-churn-analysis.git
    ```
2. Перейдите в директорию проекта:
    ```bash
    cd bank-customer-churn-analysis
    ```
3. Установите необходимые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```
4. Запустите Jupyter из **корня проекта**:
    ```bash
    jupyter notebook
    ```
5. Откройте `notebooks/bank_churn_analysis.ipynb` и выполните ячейки по порядку.

---

### Этапы обработки и моделирования (по шагам)

1. **EDA (разведывательный анализ)**  
   - Анализ распределений, зависимостей и выбросов.  
   - Выводы о группах риска (возраст, баланс, число продуктов и т.п.).

2. **Feature engineering**  
   - `BalanceSalaryRatio`, `TenureByAge`, `CreditScoreGivenAge` — для модели.  
   - `CreditScoreCat` — только для EDA и тепловых карт.

3. **Кодирование и разбиение**  
   - Кодирование `Gender` через `pd.get_dummies(..., drop_first=True)`.  
   - Стратифицированное разбиение `train/test` (`random_state=0`).

4. **Масштабирование**  
   - Для линейных моделей использовался `StandardScaler` (обучался только на `X_train`).

5. **Модели**  
   - **Логистическая регрессия** (с и без полиномиальных признаков).  
   - **Decision Tree** (базовое дерево и обрезанное: `max_depth=8`, `min_samples_leaf=10`, `criterion='entropy'`).  
   - **Random Forest** (`n_estimators=500`, `max_depth=8`, `min_samples_leaf=10`, `criterion='entropy'`).

6. **Подбор порога**  
   - Перебор `thresholds = np.arange(0.1, 1, 0.05)` для логистической регрессии с полиномиальными признаками.

---

### Метрики и результаты (ключевые)

- **Метрика:** F1-score (фокус на положительном классе — уход клиента).

| Модель | F1 на тесте |
|--------|-------------|
| Logistic Regression (базовая) | 0.494 |
| Logistic Regression + полиномы + порог **0.30** | **0.655** ← *максимальный F1* |
| Decision Tree (необрезанное) | 0.527 |
| Pruned Decision Tree | 0.651 |
| Random Forest (n=500, max_depth=8) | 0.645 |

**Вывод:**

- **Наилучший F1 (0.655)** — логистическая регрессия с полиномиальными признаками и порогом вероятности **0.30**.
- **Pruned Decision Tree (F1 = 0.651)** выбрано для финального прогноза: модель проще в интерпретации, не требует масштабирования и подбора порога, устойчива к переобучению по сравнению с необрезанным деревом.
- Для обрезанного дерева подбор порога не дал прироста — оптимальный порог ≈ 0.5, F1 остался 0.651.

---

### Пример: прогноз для конкретного клиента (Василий)

Модель: **Pruned Decision Tree**.

Входные данные:
```python
{
 'CreditScore': [601.0],
 'Gender': ['Male'],
 'Age': [42.0],
 'Tenure': [1.0],
 'Balance': [98495.72],
 'NumOfProducts': [1.0],
 'HasCrCard': [1.0],
 'IsActiveMember': [0.0],
 'EstimatedSalary': [40014.76]
}
```

**Результат:** вероятность оттока ≈ **0.800** (80%).

---

### Результаты анализа

- Наибольшая вероятность оттока наблюдается у клиентов с **низким кредитным рейтингом (`Very_Poor` и `Poor`)** и **небольшим сроком обслуживания в банке**.  
- Клиенты с рейтингом **`Good` и выше** значительно реже уходят, что подтверждает важность кредитной надёжности при удержании.  
- Категоризация кредитного рейтинга (`CreditScoreCat`) улучшила **интерпретируемость EDA** и выделила группы повышенного риска.  
- Анализ зависимости между **`Tenure`** и вероятностью ухода показал: **новые клиенты чаще прекращают обслуживание**, тогда как долгосрочные клиенты более лояльны.  
- **Логистическая регрессия с полиномиальными признаками** и порогом **0.30** достигла **наилучшего F1 = 0.655**.  
- **Обрезанное дерево решений** показало близкий результат (**F1 = 0.651**) и использовано для демонстрационного прогноза.  
- Необрезанное дерево и случайный лес демонстрировали **переобучение** или отсутствие прироста качества.

---

## 👤 Автор

**Karim** · 2026

[![GitHub](https://img.shields.io/badge/GitHub-theKerimKerimov-181717?logo=github)](https://github.com/theKerimKerimov)<br>
[![Kaggle](https://img.shields.io/badge/Kaggle-kerimkerimov-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/kerimkerimov)<br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kerim--kerimov-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kerim-kerimov-79323b400)<br>
[![LeetCode](https://img.shields.io/badge/LeetCode-KerimK-FFA116?logo=leetcode&logoColor=black)](https://leetcode.com/u/KerimK)<br>
[![Email](https://img.shields.io/badge/Email-k.kerimow%40yandex.ru-EA4335?logo=gmail&logoColor=white)](mailto:k.kerimow@yandex.ru)<br>
[![Telegram](https://img.shields.io/badge/Telegram-@theDagestani-26A5E4?logo=telegram&logoColor=white)](https://t.me/theDagestani)<br>

📍 Москва