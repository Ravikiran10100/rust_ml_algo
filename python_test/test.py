import rust_ml

X =[
    [1.0, 2.0],
    [2.0, 0.5],
    [3.0, 4.0],
]

y = [12.0, 9.0, 23.0]

model = rust_ml.LinearRegression()

model.fit(X, y)

preds = model.predict(X)

print(preds)