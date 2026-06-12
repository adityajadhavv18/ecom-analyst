import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

CUSTOMERS = [f"C{str(i).zfill(4)}" for i in range(1, 151)]
PRODUCTS = {
    "Electronics":   [f"ELEC{i:03}" for i in range(1, 11)],
    "Clothing":      [f"CLTH{i:03}" for i in range(1, 11)],
    "Home & Kitchen":[f"HOME{i:03}" for i in range(1, 11)],
    "Sports":        [f"SPRT{i:03}" for i in range(1, 11)],
    "Books":         [f"BOOK{i:03}" for i in range(1, 11)],
}
PRICE_RANGE = {
    "Electronics":    (500, 5000),
    "Clothing":       (200, 2000),
    "Home & Kitchen": (100, 3000),
    "Sports":         (150, 4000),
    "Books":          (50,  800),
}
STATUS_WEIGHTS = ["completed"] * 80 + ["returned"] * 12 + ["cancelled"] * 8

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)

rows = []
order_counter = 1

for _ in range(1000):
    customer_id  = random.choice(CUSTOMERS)
    category     = random.choice(list(PRODUCTS.keys()))
    product_id   = random.choice(PRODUCTS[category])
    quantity     = random.randint(1, 5)
    low, high    = PRICE_RANGE[category]
    unit_price   = round(random.uniform(low, high), 2)
    discount_pct = random.choice([0, 0, 0, 5, 10, 15, 20, 25])
    total_amount = round(unit_price * quantity * (1 - discount_pct / 100), 2)
    order_date   = START_DATE + timedelta(days=random.randint(0, 364))
    status       = random.choice(STATUS_WEIGHTS)
    order_id     = f"ORD{str(order_counter).zfill(5)}"
    order_counter += 1

    rows.append({
        "order_date":       order_date.strftime("%Y-%m-%d"),
        "customer_id":      customer_id,
        "order_id":         order_id,
        "product_id":       product_id,
        "product_category": category,
        "quantity":         quantity,
        "unit_price":       unit_price,
        "total_amount":     total_amount,
        "discount_percent": discount_pct,
        "order_status":     status,
    })

df = pd.DataFrame(rows)
df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv("sample_orders.csv", index=False)
print(f"Generated {len(df)} rows across {df['customer_id'].nunique()} customers")
print(df.head())