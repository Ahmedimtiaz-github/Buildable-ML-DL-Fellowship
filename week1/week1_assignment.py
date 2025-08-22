import statistics
import sys

# -------- Question 2: Mutable vs Immutable demo --------
def question2_demo():
    print("\n--- Question 2: Mutable vs Immutable demo ---\n")
    # Tuple of 3 elements
    t = (1, 2, 3)
    print("Original tuple:", t)
    try:
        t[0] = 10
    except TypeError as e:
        print("Trying to change tuple[0] raised:", type(e).__name__, "-", e)
        print("Explanation: tuples are immutable; individual elements cannot be reassigned.")

    # List of 3 elements
    lst = [1, 2, 3]
    print("\nOriginal list:", lst)
    lst[0] = 10
    print("After modifying list[0] = 10 ->", lst)
    print("Explanation: lists are mutable; elements can be changed.")

    # Dictionary update
    d = {"apple": 5, "banana": 3}
    print("\nOriginal dict:", d)
    d["banana"] = 7
    print("After updating 'banana' to 7 ->", d)
    print("Explanation: dicts are mutable; values for keys can be updated.")

    # Tuple containing sub-lists
    tt = ([1, 2], [3, 4])
    print("\nTuple with sub-lists:", tt)
    tt[0][0] = 99
    print("Modified an element inside a sub-list:", tt)
    print("Explanation: the tuple object is immutable (you cannot reassign tt[0] = ...),")
    print("but mutable objects contained inside (the lists) can be modified.")

# -------- Question 3: User Information Dictionary (validation) --------
def valid_name(s):
    return len(s.strip()) > 0

def valid_age(a):
    try:
        ai = int(a)
        return 0 < ai < 100
    except:
        return False

def valid_email(e):
    e = e.strip()
    # simple checks: contains @ and ., does not start/end with special chars, '.' after '@'
    if '@' not in e or '.' not in e:
        return False
    if e[0] in "@._-" or e[-1] in "@._-":
        return False
    try:
        at = e.index('@')
        dot_after = '.' in e[at+1:]
        return dot_after
    except ValueError:
        return False

def valid_favnum(n):
    try:
        ni = int(n)
        return 1 <= ni <= 100
    except:
        return False

def question3_user_info():
    print("\n--- Question 3: User Information Dictionary ---\n")
    while True:
        name = input("Enter your name: ").strip()
        if valid_name(name): break
        print("Name cannot be empty. Try again.")
    while True:
        age = input("Enter your age (number): ").strip()
        if valid_age(age):
            age = int(age); break
        print("Age must be a positive integer less than 100.")
    while True:
        email = input("Enter your email: ").strip()
        if valid_email(email): break
        print("Invalid email. It must contain '@' and '.' and not start/end with special characters.")
    while True:
        fav = input("Enter your favorite number (1-100): ").strip()
        if valid_favnum(fav):
            fav = int(fav); break
        print("Favorite number must be an integer between 1 and 100.")

    user = {"name": name, "age": age, "email": email, "favorite_number": fav}
    print("\nStored user dictionary:", user)
    print(f"\nWelcome {name}! Your account has been registered with email {email}.")

# -------- Question 4: Cinema Ticketing System --------
def calculate_ticket_price(age, is_student, is_weekend):
    # Validate
    if age < 0 or age > 120:
        raise ValueError("Invalid age: must be between 0 and 120")
    # base price rules
    if age < 12:
        price = 5.0
    elif 13 <= age <= 17:
        price = 8.0
    elif 18 <= age <= 59:
        price = 12.0
    else: # 60+
        price = 6.0
    # student discount: only applies if age > 12 (students above 12)
    if is_student and age > 12:
        price *= 0.8  # 20% off
    if is_weekend:
        price += 2.0
    return round(price, 2)

def question4_cinema():
    print("\n--- Question 4: Cinema Ticketing System ---\n")
    try:
        n = int(input("How many customers? ").strip())
        if n <= 0:
            print("Number of customers must be positive.")
            return
    except:
        print("Please enter a valid integer for number of customers.")
        return

    customers = []
    total = 0.0
    for i in range(1, n+1):
        print(f"\nCustomer #{i}:")
        while True:
            try:
                age = int(input("  Age: ").strip())
                if age < 0 or age > 120:
                    print("  Invalid age. Enter a realistic age (0-120).")
                    continue
                break
            except:
                print("  Please enter a valid integer for age.")
        student = input("  Is student? (yes/no): ").strip().lower() == 'yes'
        weekend = input("  Is it a weekend show? (yes/no): ").strip().lower() == 'yes'
        try:
            price = calculate_ticket_price(age, student, weekend)
        except ValueError as e:
            print("  Error:", e)
            return
        customers.append({"customer": i, "age": age, "student": student, "weekend": weekend, "price": price})
        total += price

    # Group discount if 4 or more customers
    group_discount = 0.0
    if len(customers) >= 4:
        group_discount = round(total * 0.10, 2)  # 10% discount
        total_after = round(total - group_discount, 2)
    else:
        total_after = round(total, 2)

    print("\nCustomer ticket details:")
    for c in customers:
        print(f"  Customer {c['customer']}: age={c['age']}, student={c['student']}, weekend={c['weekend']}, price=${c['price']:.2f}")

    print(f"\nTotal before discount: ${round(total,2):.2f}")
    if group_discount > 0:
        print(f"Group discount (10%): -${group_discount:.2f}")
    print(f"Total after discount: ${total_after:.2f}")

    # highest and lowest paying customers
    highest = max(customers, key=lambda x: x["price"])
    lowest = min(customers, key=lambda x: x["price"])
    print(f"Highest paying customer: Customer {highest['customer']} paying ${highest['price']:.2f}")
    print(f"Lowest paying customer: Customer {lowest['customer']} paying ${lowest['price']:.2f}")

# -------- Question 5: Weather Alert System --------
def c_to_f(c): return (c * 9/5) + 32
def c_to_k(c): return c + 273.15

def weather_alert(temp_celsius, condition, include_conversions=True):
    cond = condition.strip().lower()
    t = temp_celsius
    msg = "Normal weather conditions."
    if t < 0 and "snow" in cond:
        msg = "Heavy snow alert! Stay indoors."
    elif t > 35 and "sun" in cond:
        msg = "Heatwave warning! Stay hydrated."
    elif "rain" in cond and t < 15:
        msg = "Cold rain alert! Wear warm clothes."
    if include_conversions:
        return f"{msg} (Temp: {t}°C / {round(c_to_f(t),2)}°F / {round(c_to_k(t),2)}K)"
    else:
        return msg

def question5_weather():
    print("\n--- Question 5: Weather Alert System ---\n")
    try:
        t = float(input("Enter temperature in Celsius: ").strip())
    except:
        print("Invalid temperature.")
        return
    cond = input("Enter condition (sunny, rainy, snowy, etc.): ")
    print(weather_alert(t, cond))

# -------- Question 6: Sales Analytics (max, min, median) --------
def analyze_sales(sales_list):
    if not sales_list:
        raise ValueError("Empty sales list.")
    mx = max(sales_list)
    mn = min(sales_list)
    med = statistics.median(sales_list)
    return mx, mn, med

def question6_sales():
    print("\n--- Question 6: Sales Analytics ---\n")
    print("Enter at least 5 daily sales values (comma-separated). Example: 100,200,150,75,210")
    raw = input("Sales: ").strip()
    try:
        values = [float(x.strip()) for x in raw.split(",") if x.strip()!='']
    except:
        print("Invalid input. Use comma-separated numbers.")
        return
    if len(values) < 5:
        print("Please enter at least 5 daily sales values.")
        return
    mx, mn, med = analyze_sales(values)
    print(f"Highest sales day: {mx}")
    print(f"Lowest sales day: {mn}")
    print(f"Median sales: {med}")

# -------- Question 7: E-commerce Inventory Management --------
def update_inventory(inventory_dict, item, quantity):
    if item not in inventory_dict:
        inventory_dict[item] = 0
    new_qty = inventory_dict[item] + quantity
    if new_qty < 0:
        return False, f"Not enough stock for {item}"
    inventory_dict[item] = new_qty
    return True, inventory_dict

def question7_inventory():
    print("\n--- Question 7: E-commerce Inventory Management ---\n")
    inventory = {
        "apple": 10,
        "banana": 8,
        "chocolate": 5,
        "soap": 12,
        "notebook": 7
    }
    print("Initial inventory:", inventory)
    print("Simulate buying 3 items. For each, enter item name and quantity to buy (positive integer).")
    for i in range(3):
        item = input(f"Item #{i+1} name: ").strip()
        try:
            qty = int(input(f"Quantity of '{item}': ").strip())
        except:
            print("Invalid quantity. Skipping this item.")
            continue
        # buying means negative change
        success, result = update_inventory(inventory, item, -qty)
        if not success:
            print(result)
        else:
            print(f"Purchased {qty} of {item}.")
    print("Inventory after checkout:", inventory)
    # most and least stocked products
    most = max(inventory.items(), key=lambda x: x[1])
    least = min(inventory.items(), key=lambda x: x[1])
    print(f"Most stocked product: {most[0]} ({most[1]})")
    print(f"Least stocked product: {least[0]} ({least[1]})")

# -------- Main menu for running each question --------
def main():
    print("Week1 Assignment helper. Choose which question to run:")
    print("2 - Mutable vs Immutable demo")
    print("3 - User info validation")
    print("4 - Cinema ticketing system")
    print("5 - Weather alert system")
    print("6 - Sales analytics")
    print("7 - Inventory management")
    print("q - Quit")
    while True:
        choice = input("\nEnter choice (2/3/4/5/6/7/q): ").strip().lower()
        if choice == '2':
            question2_demo()
        elif choice == '3':
            question3_user_info()
        elif choice == '4':
            question4_cinema()
        elif choice == '5':
            question5_weather()
        elif choice == '6':
            question6_sales()
        elif choice == '7':
            question7_inventory()
        elif choice == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()