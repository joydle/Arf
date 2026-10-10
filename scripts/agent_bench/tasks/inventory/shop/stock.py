from shop.parse import parse_line


def load(text):
    """One item per non-empty line; a repeated name adds to its quantity."""
    items = {}
    for line in text.splitlines():
        name, qty, price = parse_line(line)
        items[name] = (qty, price)
    return items


def total_value(items):
    return sum(qty for qty, price in items.values())
