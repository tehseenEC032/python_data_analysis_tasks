import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

project_path = r'C:\Users\91720\OneDrive\Desktop\project'

# TASK 1: Data Analysis with Pandas and Matplotlib
df_sales = pd.read_csv(f'{project_path}\\sales_data.csv')
print("Data loaded successfully:")
print(df_sales.head())
average_sales = df_sales['Sales'].mean()
print(f"\nAverage sales across all products: ${average_sales:.2f}")
product_sales = df_sales.groupby('Product')['Sales'].sum()
product_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=0)
plt.show()

# TASK 2: Linear Regression Model
df_houses = pd.read_csv(f'{project_path}\\house_prices.csv')
X = df_houses[['size_sq_ft', 'number_of_rooms']]
y = df_houses['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
new_house = np.array([[1750, 4]])
predicted_price = model.predict(new_house)
print(f"Predicted price for a new house: ${predicted_price[0]:.2f}")

# TASK 3: Matrix Operations Tool
def get_matrix_input(prompt):
    print(prompt)
    while True:
        rows_str = input("Enter number of rows: ")
        cols_str = input("Enter number of columns: ")
        if not rows_str.isdigit() or not cols_str.isdigit():
            print("Invalid input")
            continue
        rows = int(rows_str)
        cols = int(cols_str)
        print("Enter elements row by row, separated by spaces:")
        matrix = []
        is_valid = True
        for i in range(rows):
            row_str = input(f"Row {i+1}: ")
            row_elements = row_str.split()
            if len(row_elements) != cols:
                print("The number of elements does not match the number of columns")
                is_valid = False
                break
            try:
                row = [int(x) for x in row_elements]
                matrix.append(row)
            except ValueError:
                print("Invalid input")
                is_valid = False
                break
        if is_valid:
            return np.array(matrix)
while True:
    print("\n--- Matrix Operations Menu ---")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Transpose")
    print("5. Determinant")
    print("6. Exit")
    choice = input("Enter your choice (1-6): ")
    if choice == '1':
        A = get_matrix_input("Enter Matrix A:")
        B = get_matrix_input("Enter Matrix B:")
        # Check if matrices have the same dimensions before adding
        if A.shape == B.shape:
            print("\nResult of A + B:")
            print(np.add(A, B))
        else:
            print("\n Matrices must have the same dimensions for addition.")
    elif choice == '2':
        A = get_matrix_input("Enter Matrix A:")
        B = get_matrix_input("Enter Matrix B:")
        if A.shape == B.shape:
            print("\nResult of A - B:")
            print(np.subtract(A, B))
        else:
            print("\n Matrices must have the same dimensions for subtraction.")
    elif choice == '3':
        A = get_matrix_input("Enter Matrix A:")
        B = get_matrix_input("Enter Matrix B:")
        # Check multiplication compatibility: A's columns must equal B's rows
        if A.shape[1] == B.shape[0]:
            print("\nResult of A * B:")
            print(np.dot(A, B))
        else:
            print("\n The number of columns in the first matrix must equal the number of rows in the second matrix for multiplication.")

    elif choice == '4':
        A = get_matrix_input("Enter Matrix A:")
        print("\nResult of Transpose(A):")
        print(np.transpose(A))

    elif choice == '5':
        A = get_matrix_input("Enter Matrix A (must be a square matrix):")

        if A.shape[0] == A.shape[1]:
            print("\nResult of Determinant(A):")
            print(np.linalg.det(A))
        else:
            print("\n Determinant can only be calculated for a square matrix.")
            
    elif choice == '6':
        print("Exit")
        break
    else:
        print("Invalid choice. Please enter a number from 1 to 6.")
