"""
Day 1: Python + NumPy Practice Exercises
Complete all 10 exercises to master Python basics
"""

exercises = [
    {
        "id": 1,
        "title": "Distance Calculator",
        "description": "Calculate total distance and average from multiple runs",
        "difficulty": "Easy",
        "instructions": """Given three run distances (5.2 km, 10.5 km, 3.8 km):
1. Calculate the total distance
2. Calculate the average distance
3. Print both results""",
        "starter_code": """# Exercise 1: Distance Calculator
run1 = 5.2
run2 = 10.5
run3 = 3.8

# Calculate total distance
total = 0  # TODO: Calculate sum

# Calculate average distance
average = 0  # TODO: Calculate average

print(f"Total distance: {total} km")
print(f"Average distance: {average} km")""",
        "solution": """# Exercise 1: Distance Calculator - Solution
run1 = 5.2
run2 = 10.5
run3 = 3.8

# Calculate total distance
total = run1 + run2 + run3

# Calculate average distance
average = total / 3

print(f"Total distance: {total} km")
print(f"Average distance: {average:.2f} km")""",
        "expected_output": """Total distance: 19.5 km
Average distance: 6.50 km""",
        "hints": [
            "Add all three distances together",
            "Divide the total by 3 to get average",
            "Use {:.2f} for 2 decimal places"
        ],
        "explanation": """This exercise teaches basic arithmetic operations and variable usage.
Total = sum of all values
Average = total / number of values"""
    },
    {
        "id": 2,
        "title": "Name Formatter",
        "description": "Format a name to title case",
        "difficulty": "Easy",
        "instructions": """Convert a mixed-case name to proper title case.
Input: "AKHIL reddy"
Output: "Akhil Reddy" """,
        "starter_code": """# Exercise 2: Name Formatter
name = "AKHIL reddy"

# Convert to title case
formatted_name = ""  # TODO: Use .title() method

print(f"Formatted name: {formatted_name}")""",
        "solution": """# Exercise 2: Name Formatter - Solution
name = "AKHIL reddy"

# Convert to title case
formatted_name = name.title()

print(f"Formatted name: {formatted_name}")""",
        "expected_output": """Formatted name: Akhil Reddy""",
        "hints": [
            "Python has a built-in string method for title case",
            "Try: string.title()",
            "This capitalizes the first letter of each word"
        ],
        "explanation": """The .title() method converts a string to title case.
It capitalizes the first letter of each word."""
    },
    {
        "id": 3,
        "title": "Pace Converter",
        "description": "Convert running pace from min/km to min/mile",
        "difficulty": "Easy",
        "instructions": """Convert pace from minutes per kilometer to minutes per mile.
1 mile = 1.60934 km
Input: 5.5 min/km
Output: pace in min/mile""",
        "starter_code": """# Exercise 3: Pace Converter
pace_min_km = 5.5
km_per_mile = 1.60934

# Convert to min/mile
pace_min_mile = 0  # TODO: Calculate conversion

print(f"Pace: {pace_min_km} min/km = {pace_min_mile:.2f} min/mile")""",
        "solution": """# Exercise 3: Pace Converter - Solution
pace_min_km = 5.5
km_per_mile = 1.60934

# Convert to min/mile
pace_min_mile = pace_min_km * km_per_mile

print(f"Pace: {pace_min_km} min/km = {pace_min_mile:.2f} min/mile")""",
        "expected_output": """Pace: 5.5 min/km = 8.85 min/mile""",
        "hints": [
            "If 1 mile = 1.60934 km, then running 1 mile takes longer",
            "Multiply pace by the conversion factor",
            "pace_min_mile = pace_min_km * 1.60934"
        ],
        "explanation": """To convert pace from min/km to min/mile:
Multiply by the km_per_mile factor (1.60934)"""
    },
    {
        "id": 4,
        "title": "Challenge Eligibility",
        "description": "Check if user qualifies for a challenge",
        "difficulty": "Easy",
        "instructions": """Check if a user qualifies based on:
- Age must be >= 18
- Distance ran must be >= 5.0 km
Return True if both conditions met, False otherwise""",
        "starter_code": """# Exercise 4: Challenge Eligibility
age = 25
distance_ran = 6.2

# Check eligibility
qualifies = False  # TODO: Use AND logic

print(f"Qualifies for challenge: {qualifies}")""",
        "solution": """# Exercise 4: Challenge Eligibility - Solution
age = 25
distance_ran = 6.2

# Check eligibility
qualifies = age >= 18 and distance_ran >= 5.0

print(f"Qualifies for challenge: {qualifies}")""",
        "expected_output": """Qualifies for challenge: True""",
        "hints": [
            "Use the 'and' operator to combine conditions",
            "Both conditions must be True",
            "Syntax: condition1 and condition2"
        ],
        "explanation": """Boolean logic with AND operator:
True and True = True
True and False = False
Both conditions must be satisfied."""
    },
    {
        "id": 5,
        "title": "Find Fastest Pace",
        "description": "Find the minimum pace from a list",
        "difficulty": "Easy",
        "instructions": """Given a list of paces in min/km, find the fastest (minimum) pace.""",
        "starter_code": """# Exercise 5: Find Fastest Pace
paces = [5.5, 4.8, 6.2, 5.0, 4.5]

# Find fastest (minimum) pace
fastest = 0  # TODO: Use min() function

print(f"Fastest pace: {fastest} min/km")""",
        "solution": """# Exercise 5: Find Fastest Pace - Solution
paces = [5.5, 4.8, 6.2, 5.0, 4.5]

# Find fastest (minimum) pace
fastest = min(paces)

print(f"Fastest pace: {fastest} min/km")""",
        "expected_output": """Fastest pace: 4.5 min/km""",
        "hints": [
            "Python has a built-in min() function",
            "Pass the list to min()",
            "Fastest pace = lowest number"
        ],
        "explanation": """The min() function returns the smallest value in a sequence.
min([5.5, 4.8, 6.2]) returns 4.8"""
    },
    {
        "id": 6,
        "title": "User Info Extractor",
        "description": "Extract and format data from a dictionary",
        "difficulty": "Medium",
        "instructions": """Extract values from a user dictionary and format a message.
Format: "Name, Age, from City ran Distance km" """,
        "starter_code": """# Exercise 6: User Info Extractor
user = {"name": "Akhil", "age": 25, "city": "Hyderabad", "distance": 42.5}

# Extract values and format message
message = ""  # TODO: Use f-string or .format()

print(message)""",
        "solution": """# Exercise 6: User Info Extractor - Solution
user = {"name": "Akhil", "age": 25, "city": "Hyderabad", "distance": 42.5}

# Extract values and format message
message = f"{user['name']}, {user['age']}, from {user['city']} ran {user['distance']} km"

print(message)""",
        "expected_output": """Akhil, 25, from Hyderabad ran 42.5 km""",
        "hints": [
            "Access dictionary values with user['key']",
            "Use f-string: f\"text {variable}\"",
            "Combine all values with commas and text"
        ],
        "explanation": """Dictionary access: dict['key'] returns the value.
F-strings allow embedding variables in strings."""
    },
    {
        "id": 7,
        "title": "Fitness Level Calculator",
        "description": "Assign fitness level based on distance",
        "difficulty": "Medium",
        "instructions": """Assign fitness level based on distance:
- distance < 5: "Beginner"
- 5 <= distance < 10: "Intermediate"
- 10 <= distance < 20: "Advanced"
- distance >= 20: "Expert" """,
        "starter_code": """# Exercise 7: Fitness Level Calculator
distance = 12.5

# Assign fitness level
level = ""  # TODO: Use if/elif/else

print(f"Fitness level: {level}")""",
        "solution": """# Exercise 7: Fitness Level Calculator - Solution
distance = 12.5

# Assign fitness level
if distance < 5:
    level = "Beginner"
elif distance < 10:
    level = "Intermediate"
elif distance < 20:
    level = "Advanced"
else:
    level = "Expert"

print(f"Fitness level: {level}")""",
        "expected_output": """Fitness level: Advanced""",
        "hints": [
            "Use if/elif/else for multiple conditions",
            "Check from smallest to largest",
            "elif means 'else if'"
        ],
        "explanation": """Conditional statements execute different code based on conditions.
elif allows checking multiple conditions in sequence."""
    },
    {
        "id": 8,
        "title": "Leaderboard Formatter",
        "description": "Format time from seconds to MM:SS",
        "difficulty": "Medium",
        "instructions": """Convert time in seconds to MM:SS format.
Input: 1825 seconds
Output: "#1 - Akhil - 30:25" """,
        "starter_code": """# Exercise 8: Leaderboard Formatter
rank = 1
name = "Akhil"
time_seconds = 1825

# Convert to MM:SS
minutes = 0  # TODO: Use // for integer division
seconds = 0  # TODO: Use % for remainder

formatted = ""  # TODO: Format as "#Rank - Name - MM:SS"

print(formatted)""",
        "solution": """# Exercise 8: Leaderboard Formatter - Solution
rank = 1
name = "Akhil"
time_seconds = 1825

# Convert to MM:SS
minutes = time_seconds // 60
seconds = time_seconds % 60

formatted = f"#{rank} - {name} - {minutes}:{seconds:02d}"

print(formatted)""",
        "expected_output": """#1 - Akhil - 30:25""",
        "hints": [
            "// is floor division (integer result)",
            "% is modulus (remainder)",
            "{:02d} formats number with leading zero"
        ],
        "explanation": """Floor division // gives whole minutes.
Modulus % gives remaining seconds.
:02d formats with 2 digits, padding with 0."""
    },
    {
        "id": 9,
        "title": "Common Cities Finder",
        "description": "Find intersection of two sets",
        "difficulty": "Medium",
        "instructions": """Find cities that appear in both users' visited cities lists.""",
        "starter_code": """# Exercise 9: Common Cities Finder
user1_cities = {"Hyderabad", "Bangalore", "Mumbai"}
user2_cities = {"Delhi", "Bangalore", "Hyderabad"}

# Find common cities
common = set()  # TODO: Use set intersection

print(f"Common cities: {common}")""",
        "solution": """# Exercise 9: Common Cities Finder - Solution
user1_cities = {"Hyderabad", "Bangalore", "Mumbai"}
user2_cities = {"Delhi", "Bangalore", "Hyderabad"}

# Find common cities
common = user1_cities & user2_cities
# OR: common = user1_cities.intersection(user2_cities)

print(f"Common cities: {common}")""",
        "expected_output": """Common cities: {'Bangalore', 'Hyderabad'}""",
        "hints": [
            "Use & operator for set intersection",
            "Or use .intersection() method",
            "Both give the same result"
        ],
        "explanation": """Set intersection finds elements in both sets.
& operator or .intersection() method both work."""
    },
    {
        "id": 10,
        "title": "GPS Distance Calculator",
        "description": "Calculate distance between GPS coordinates",
        "difficulty": "Hard",
        "instructions": """Calculate approximate distance between two GPS points.
Formula: distance = sqrt((lat2-lat1)² + (lon2-lon1)²) * 111 km""",
        "starter_code": """# Exercise 10: GPS Distance Calculator
import math

start = (17.385, 78.486)  # Hyderabad
end = (17.400, 78.500)

# Unpack tuples
lat1, lon1 = start
lat2, lon2 = end

# Calculate distance
distance = 0  # TODO: Use formula

print(f"Distance: {distance:.2f} km")""",
        "solution": """# Exercise 10: GPS Distance Calculator - Solution
import math

start = (17.385, 78.486)  # Hyderabad
end = (17.400, 78.500)

# Unpack tuples
lat1, lon1 = start
lat2, lon2 = end

# Calculate distance
lat_diff = lat2 - lat1
lon_diff = lon2 - lon1
distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111

print(f"Distance: {distance:.2f} km")""",
        "expected_output": """Distance: 2.28 km""",
        "hints": [
            "Use math.sqrt() for square root",
            "** is the exponentiation operator",
            "Multiply by 111 to convert degrees to km (approximation)"
        ],
        "explanation": """This uses the Pythagorean theorem for approximate distance.
Real GPS uses Haversine formula for accuracy on Earth's sphere."""
    }
]
