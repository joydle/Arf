def parse_line(line):
    """'name, qty, price' -> (name, int qty, float price); surrounding spaces ignored."""
    name, qty, price = line.split(",")
    return name, int(qty), float(price)
