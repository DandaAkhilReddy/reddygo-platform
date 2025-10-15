# Week 1, Day 1: Python Basics - Variables, Data Types & Operators

**Learning Time:** 2-3 hours
**Difficulty:** Beginner
**Goal:** Master Python fundamentals and build your first ReddyGo utility

---

## Learning Objectives

By the end of today, you will:
- ✅ Understand Python variables and naming conventions
- ✅ Master all Python data types (int, float, str, bool, list, tuple, dict, set)
- ✅ Use arithmetic, comparison, and logical operators
- ✅ Write conditional statements (if/elif/else)
- ✅ Build a distance calculator for ReddyGo

---

## Hour 1: Theory & Concepts

### 1. Variables

Variables store data in memory. Python is dynamically typed.

```python
# Variable assignment
name = "ReddyGo"
users_count = 1000
is_active = True

# Multiple assignment
x, y, z = 1, 2, 3

# Swapping variables
a, b = 10, 20
a, b = b, a  # Now a=20, b=10
```

**Naming Rules:**
- Start with letter or underscore: `user_name`, `_id`
- Can contain letters, numbers, underscores: `user123`
- Case-sensitive: `Name` ≠ `name`
- Avoid Python keywords: `class`, `for`, `if`, etc.

**Conventions:**
- Use snake_case for variables: `user_distance`
- Use UPPER_CASE for constants: `MAX_DISTANCE`

### 2. Data Types

#### Numbers
```python
# Integer
age = 25
distance_meters = 5000

# Float
pace_min_per_km = 5.5
latitude = 37.7749

# Complex (rarely used)
z = 3 + 4j
```

#### Strings
```python
# Single or double quotes
name = 'Akhil'
city = "Hyderabad"

# Multi-line strings
description = """
ReddyGo is a fitness app
for competitive runners
"""

# String operations
first_name = "Akhil"
last_name = "Reddy"
full_name = first_name + " " + last_name  # Concatenation
repeated = "Run! " * 3  # "Run! Run! Run! "

# String methods
text = "reddygo fitness"
print(text.upper())  # "REDDYGO FITNESS"
print(text.title())  # "Reddygo Fitness"
print(text.replace("fitness", "platform"))  # "reddygo platform"

# String indexing and slicing
word = "Python"
print(word[0])      # 'P' (first character)
print(word[-1])     # 'n' (last character)
print(word[0:3])    # 'Pyt' (index 0 to 2)
print(word[2:])     # 'thon' (index 2 to end)

# f-strings (formatted strings) - Best practice!
name = "Akhil"
distance = 5.2
message = f"{name} ran {distance} km today"
print(message)  # "Akhil ran 5.2 km today"
```

#### Boolean
```python
is_running = True
has_finished = False

# Boolean operations
print(True and False)   # False
print(True or False)    # True
print(not True)         # False
```

#### Collections

**List** - Ordered, mutable, allows duplicates
```python
distances = [5.0, 10.2, 3.5, 8.1]
users = ["Akhil", "Vamshi", "Prasad"]

# Accessing elements
print(distances[0])   # 5.0
print(distances[-1])  # 8.1 (last element)

# Modifying
distances[0] = 6.0  # Change first element
distances.append(12.5)  # Add to end
distances.insert(0, 2.5)  # Insert at position 0
distances.remove(3.5)  # Remove specific value
popped = distances.pop()  # Remove and return last element

# List methods
print(len(distances))  # Length
print(sum(distances))  # Sum
print(max(distances))  # Maximum
print(min(distances))  # Minimum
```

**Tuple** - Ordered, immutable
```python
coordinates = (37.7749, -122.4194)  # Latitude, longitude
rgb_color = (255, 128, 0)

# Accessing
lat = coordinates[0]
lon = coordinates[1]

# Tuple unpacking
lat, lon = coordinates

# Tuples are immutable (cannot change)
# coordinates[0] = 38.0  # This would cause an error!
```

**Dictionary** - Key-value pairs, unordered
```python
user = {
    "id": 1001,
    "name": "Akhil",
    "distance_km": 42.5,
    "city": "Hyderabad"
}

# Accessing values
print(user["name"])  # "Akhil"
print(user.get("email", "Not provided"))  # Safe access with default

# Modifying
user["distance_km"] = 45.0  # Update
user["email"] = "akhil@reddygo.com"  # Add new key

# Dictionary methods
print(user.keys())    # All keys
print(user.values())  # All values
print(user.items())   # Key-value pairs
```

**Set** - Unordered, unique elements
```python
visited_cities = {"Hyderabad", "Bangalore", "Mumbai"}
completed_challenges = {101, 102, 103, 101}  # Duplicate 101 ignored

# Set operations
cities_a = {"Hyderabad", "Bangalore"}
cities_b = {"Bangalore", "Mumbai"}

print(cities_a | cities_b)  # Union: all cities
print(cities_a & cities_b)  # Intersection: common cities
print(cities_a - cities_b)  # Difference: only in A
```

### 3. Operators

#### Arithmetic Operators
```python
# Basic math
a = 10
b = 3

print(a + b)   # 13 (addition)
print(a - b)   # 7 (subtraction)
print(a * b)   # 30 (multiplication)
print(a / b)   # 3.333... (division - always float)
print(a // b)  # 3 (floor division - integer)
print(a % b)   # 1 (modulus - remainder)
print(a ** b)  # 1000 (exponentiation)

# Practical example
total_seconds = 3665
hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60
```

#### Comparison Operators
```python
x = 10
y = 20

print(x == y)   # False (equal to)
print(x != y)   # True (not equal to)
print(x > y)    # False (greater than)
print(x < y)    # True (less than)
print(x >= 10)  # True (greater than or equal)
print(x <= 20)  # True (less than or equal)
```

#### Logical Operators
```python
age = 25
distance = 5.0

# AND - both must be True
can_join = age >= 18 and distance >= 3.0
print(can_join)  # True

# OR - at least one must be True
is_eligible = age >= 18 or distance >= 10.0
print(is_eligible)  # True

# NOT - reverse the boolean
is_minor = not (age >= 18)
print(is_minor)  # False
```

### 4. Control Flow (if/elif/else)

```python
distance = 5.2

if distance < 5:
    level = "Beginner"
elif distance < 10:
    level = "Intermediate"
elif distance < 20:
    level = "Advanced"
else:
    level = "Expert"

print(f"You ran {distance} km - {level} level!")
```

**Ternary operator** (one-line if/else):
```python
age = 17
status = "Adult" if age >= 18 else "Minor"
```

---

## Hour 2: Practice Exercises

Open `exercises.py` and solve the 10 practice problems.

---

## Hour 3: ReddyGo Project

Open `reddygo_validator.py` and build a user input validator for the ReddyGo platform.

---

## Resources

**Video Tutorials** (Watch 1-2):
- Python Variables & Data Types: https://www.youtube.com/watch?v=OH86oLzVzzw
- Python Operators: https://www.youtube.com/watch?v=v5MR5JnKcZI
- Conditional Statements: https://www.youtube.com/watch?v=DZwmZ8Usvnk

**Reading:**
- Python Official Tutorial: https://docs.python.org/3/tutorial/introduction.html
- Real Python - Variables: https://realpython.com/python-variables/

---

## Success Criteria

- ✅ Complete all 10 practice exercises
- ✅ Build working ReddyGo validator
- ✅ Test your code with different inputs
- ✅ Document your learnings in `notes.md`

---

**Next:** Day 2 - More control flow (loops) and string manipulation
