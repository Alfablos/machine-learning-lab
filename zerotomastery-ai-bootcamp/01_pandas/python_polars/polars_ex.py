import polars as pl
import matplotlib.pyplot as plt

car_sales = pl.read_csv('car-sales.csv')
print(car_sales)


car_sales = car_sales.select(
    # Lowercase for values and column names
    pl.col('Make', 'Colour').str.to_lowercase().name.to_lowercase(),
    pl.col('Doors').name.to_lowercase(),
    pl.col('Odometer (KM)'),
    pl.col('Price')
        .str.replace_all(r"\$|,", '')
        .str.to_decimal()
        .name.to_lowercase()
)

car_sales = car_sales.rename({'Odometer (KM)': 'odometer_km'})

print(car_sales)