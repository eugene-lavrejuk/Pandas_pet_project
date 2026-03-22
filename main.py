import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")

melb_data = pd.read_csv('https://raw.githubusercontent.com/esabunor/MLWorkspace/master/melb_data.csv') # Загружаем данные из интернета

print("=== 1. ИНФОРМАЦИЯ О ДАТАСЕТЕ ===")   # 1. Общая информация о датасете
print()
print(melb_data.info())

print("\n=== 1.1 ПРОПУСКИ ===")   # 1.1. Количество пропусков в каждой колонке
print()
print(melb_data.isnull().sum())

print("\n=== 1.2. ОБЩАЯ СТАТИСТИКА: ===") # 1.2. Статистика по числовым колонкам
print()
print(melb_data.describe())

categorical_cols = ['Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'] # 1.3. Название улиц, районов и тд.
for col in categorical_cols:
    if col in melb_data.columns:
        print(f"\n=== 1.3. {col}: уникальные значения ===")
        print(melb_data[col].value_counts())

print("\n=== 2. ОЧИСТКА: ===")   # 2. ИСПРАВЛЕНИЯ И ОЧИСТКА ДАННЫХ
print()
print("Размер до:", melb_data.shape)
print("Пропуски всего:", melb_data.isnull().sum().sum())

print("\n=== 2.1. Исправление опечаток ===")  # 2.1. Исправление опечаток
print()
melb_data = melb_data.rename(columns={'Lattitude': 'Latitude', 'Longtitude': 'Longitude'})
print("ДО → Lattitude/Longtitude ПОСЛЕ → Latitude/Longitude")

# 2.2. Удаление ненужных колонок
print("\n=== 2.2. Удаление ненужных колонок ===")  # 2.2. Удаление, которые не являются полезными для анализа
print()
cols_to_drop = ['Unnamed: 0', 'Address', 'Bedroom2']
melb_data = melb_data.drop(columns=cols_to_drop)
print("Удалены:", cols_to_drop)
print("Колонок осталось:", len(melb_data.columns))

print("\n=== 2.3. Делаем удобный формат даты ===")  # 2.3. Превращаем колонку Date в удобный формат даты и добавляем месяц/год для анализа трендов
print()
print('Превращаем колонку Date в удобный формат даты и добавляем месяц/год для анализа трендов.')
melb_data['Date'] = pd.to_datetime(melb_data['Date'], format='%d/%m/%Y')
melb_data['Month'] = melb_data['Date'].dt.month
melb_data['Year'] = melb_data['Date'].dt.year
print("Date → datetime + Month/Year добавлены")


print("\n=== 2.4. Замена пустых значений в Landsize, BuildingArea, YearBuilt на медиану ===")
print()
print('В Landsize, BuildingArea, YearBuilt большое кол-во NaN (~40%), чтобы при анализе данных мы не потеряли почти половину строк заменяем пустые значения с помощью fillna.')
for col in ['Landsize', 'BuildingArea']:
    melb_data[col] = melb_data.groupby('Type')[col].transform(lambda x: x.fillna(x.median()))
melb_data['YearBuilt'] = melb_data['YearBuilt'].fillna(melb_data['YearBuilt'].median())
print("Заполнены: Landsize/BuildingArea (по Type), YearBuilt (медиана)")

print("\n=== 2.5. Drop строк с NaN в CouncilArea, Latitude, Longitude ===")
print()
print('Здесь не такое кол-во строк по сравнению с Landsize, BuildingArea, YearBuilt, чтобы не искажать данные еще сильнее, было принято решение их удалить.')
key_cols = ['CouncilArea', 'Latitude', 'Longitude']
melb_data.dropna(subset=key_cols, inplace=True)
print("Удалены строки с NaN в:", key_cols)


print("\n=== 2.6. Фильтр выбросов ===") # Удаляем сильно выбивающиеся значения, в данном случае это ~50–100 строк с виллами/особнями, которые могут исказить статистику.
print()
print('Создадим фильтр, для корректной оценки недвижимости.')
melb_data = melb_data[(melb_data['Rooms'] <= 10) & (melb_data['Price'] < 6000000)]
print("Фильтр выбросов: Rooms <=10, Price <6млн")


print("\n=== 2.7. Исправление типа данных ===") # 2.7. Исходный "Car" - float, что не совсем корректно.
print()
print('Исходный "Car" - float, что не совсем корректно. ')
melb_data['Car'] = melb_data['Car'].astype('int64')
print("Car → int64")

print("\n=== ОЧИСТКА ИТОГ: ===")
print("Размер после:", melb_data.shape)
print("Пропуски всего:", melb_data.isnull().sum().sum())
print("\nОБЩАЯ СТАТИСТИКА СТОИМОСТИ ПОСЛЕ ОЧИСТКИ И ЗАМЕНЫ ПУСТЫХ ЗНАЧЕНИЙ:")
print(melb_data['Price'].describe().round(0))

print("\n=== 3. СОЗДАНИЕ НОВЫХ КОЛОНОК ===")
print()
# 3.1. Добавляем колонку "возраст дома"
print("\n=== 3.1. Добавляем колонку 'возраст дома' ===")
current_year = datetime.now().year
melb_data['Age'] = current_year - melb_data['YearBuilt']
print("Добавлена колонка Age (возраст на", current_year, "год, ср.:", melb_data['Age'].mean().round(0), ")")

# 3.2. Добавляем "цена за квадратный метр" (для сравнения объектов, с np.nan если площадь 0)
print("\n=== 3.2. Добавляем колонку 'цена за квадратный метр' ===")
melb_data['Price_per_sqm'] = melb_data.apply(lambda row: row['Price'] / row['BuildingArea'] if row['BuildingArea'] > 0 else np.nan, axis=1)
print("Добавлена колонка Price_per_sqm (цена за м², ср.:", melb_data['Price_per_sqm'].mean().round(0), ")")

print("\n=== 3.3. СРЕДНЯЯ ЦЕНА ПО ТИПУ ДОМА ===")
price_by_type = melb_data.groupby('Type')['Price'].agg(['mean', 'median', 'count']).round(0)
print(price_by_type)
print("Insight: Дома (h) самые дорогие в среднем (~1.3 млн AUD)")

print("\n=== 3.4. СРЕДНЯЯ ЦЕНА ПО РЕГИОНАМ (ТОП-5 ДОРОГИХ) ===")
price_by_region = melb_data.groupby('Regionname')['Price'].agg(['mean', 'count']).round(0).sort_values('mean', ascending=False).head()
print(price_by_region)

print("\n=== 3.5. ТОП-10 РИЕЛТОРОВ ПО СУММЕ ПРОДАЖ ===")
top_sellers = melb_data.groupby('SellerG')['Price'].sum().round(0).sort_values(ascending=False).head(10)
print(top_sellers)

print("\n=== 3.6. ЦЕНА ЗА М² ПО ТИПУ ДОМА ===")
sqm_by_type = melb_data.groupby('Type')['Price_per_sqm'].agg(['mean', 'median']).round(0)
print(sqm_by_type)

print("\n=== 3.7. ФАКТОРЫ, КОТОРЫЕ ВЛИЯЮТ НА ЦЕНУ ===")
num_cols = ['Rooms', 'Bathroom', 'Car', 'Distance', 'Landsize', 'BuildingArea', 'Age', 'Propertycount', 'Price_per_sqm']
corr_df = melb_data[num_cols + ['Price']].corr()
corr_price = corr_df['Price'].sort_values(ascending=False).round(3)
print(corr_price)
print("Rooms (+0.55) и BuildingArea (+0.36) сильно поднимают цену. Distance (-0.28) — дальше дешевле. Age (-0.10) — старое чуть дешевле")

# 4. ВИЗУАЛИЗАЦИЯ
print("\n")
save_png = False
while True:
    user_input = input("Сохранить PNG после показа графиков? (y/n, да/нет, Enter=да): ").lower().strip()
    if user_input in ['', 'y', 'yes', 'да', 'д']:  # ДА
        save_png = True
        print("\n")
        print("Выбрано: Сохранить PNG!")
        break
    elif user_input in ['n', 'no', 'нет', 'н']:  # НЕТ
        save_png = False
        print("\n")
        print("Выбрано: PNG пропустить.")
        break
    else:
        print("\n")
        print("Неверный ввод! Введите y/n, да/нет или Enter (да по умолчанию).")
fig = plt.figure(figsize=(16, 12))


print("\n=== ИДЕТ ВИЗУАЛИЗАЦИЯ ГРАФИКОВ ===")

# 4.1. Средняя цена по типу дома
plt.subplot(2, 3, 1)
sns.barplot(data=melb_data, x='Type', y='Price', hue='Type', palette='husl', legend=False)
plt.title('Средняя цена по типу дома')
plt.xlabel('ТИП')
plt.ylabel('Цена (млн AUD)')
plt.ylim(0, 1_500_000)

# 4.2. Топ-5 дорогих регионов
plt.subplot(2, 3, 2)
top_regions = price_by_region.head()
ax = sns.barplot(x=top_regions.index, y=top_regions['mean'], hue=top_regions.index, palette='viridis', legend=False)
plt.xticks(rotation=45, ha='right')
plt.title('Топ-5 дорогих регионов')
plt.ylabel('Средняя цена (AUD)')
plt.ylim(0, 1_500_000)
plt.xlabel(' ')

# 4.3. Тренд по месяцам
plt.subplot(2, 3, 5)
monthly_avg = melb_data.groupby('Month')['Price'].mean()
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', color='green')
plt.title('Тренд цен по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Средняя цена')

# 4.4. Карта цен
plt.subplot(2, 3, 6)
sizes = (melb_data['Price'] / melb_data['Price'].max()) * 100
scatter = plt.scatter(melb_data['Longitude'], melb_data['Latitude'],
                      c=melb_data['Price'] / 1_000_000, s=sizes.clip(20, 150),
                      cmap='Reds', alpha=0.7, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Цена (млн AUD)')
plt.title('Цена в зависимости от расположения')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.tight_layout()

if save_png:
    fig.savefig('Melbourne_GRAPH.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("PNG СОХРАНЁН: Melbourne_GRAPH.png")
else:
    plt.show()
    print("PNG пропущен (графики только показаны).")

# 5. ЭКСПОРТ

print("\n")
save_file = False
while True:
    user_export_input = input("Вы хотите экспортировать файлы для EXCEL или POWER BI ? (y/n, да/нет, Enter=да): ").lower().strip()
    if user_export_input in ['', 'y', 'yes', 'да', 'д']:  # ДА
        save_file = True
        print("\n")
        print("Выбрано: Экспортировать файлы.")
        break
    elif user_export_input in ['n', 'no', 'нет', 'н']:  # НЕТ
        save_file = False
        print("\n")
        print("Выбрано: не экспортировать файлы.")
        break
    else:
        print("\n")
        print("Неверный ввод! Введите y/n, да/нет или Enter (да по умолчанию).")

if save_file:
    while True:
        user_export_input = input("Экспортировать:\n1. EXCEL\n2. POWER BI\n3. EXCEL И POWER BI\n0. ВЫХОД\n(Введите 1, 2, 3 или 0): ").lower().strip()
        if user_export_input == '1':  # EXCEL
            melb_data.to_excel('Melbourne_Property_Data.xlsx', index=False)
            print("\n")
            print("Выбрано: Экспортировать файл для EXCEL.")
            print("Файл создан:")
            print("   • 'Melbourne_Property_Data.xlsx' ")
        elif user_export_input == '2':  # POWER BI
            type_export_file = True
            melb_data.to_csv('Melbourne_Property_Data.csv', index=False)
            print("\n")
            print("Выбрано: Экспортировать файл для POWER BI.")
            print("Файл создан:")
            print("   • 'Melbourne_Property_Data.csv' ")
        elif user_export_input == '3':  # EXCEL И POWER BI
            type_export_file = True
            melb_data.to_excel('Melbourne_Property_Data.xlsx', index=False)
            melb_data.to_csv('Melbourne_Property_Data.csv', index=False)
            print("\n")
            print("Выбрано: Экспортировать файл для EXCEL и POWER BI.")
            print("Файлы созданы:")
            print("   • 'Melbourne_Property_Data.xlsx' ")
            print("   • 'Melbourne_Property_Data.csv' ")
        elif user_export_input == '0':
            print("\nВыход из экспорта.")
            break
        else:
            print("\n")
            print("Неверный ввод! Введите 1, 2 или 3 (да по умолчанию).")
            continue
        again = input("\nХотите ещё экспортировать? (y/n, да/нет, Enter=да): ").lower().strip()
        if again not in ['', 'y', 'yes', 'да', 'д']:
            print("\nВыход из экспорта.")
            break

print('Конец выполнения программы')
