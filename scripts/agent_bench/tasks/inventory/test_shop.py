from shop.parse import parse_line
from shop.stock import load, total_value
from shop.report import low_stock

assert parse_line(" apple , 3 , 0.5 ") == ("apple", 3, 0.5), "parse strips spaces"
items = load("apple, 3, 0.5\n\npear, 1, 2.0\napple, 2, 0.5\nfig, 10, 1.0\n")
assert items["apple"] == (5, 0.5), "repeated names add quantity"
assert abs(total_value(items) - 14.5) < 1e-9, "total value is qty * price"
assert low_stock(items, 5) == ["pear"], "low stock"
assert low_stock(items, 6) == ["apple", "pear"], "low stock sorted"
print("OK")
