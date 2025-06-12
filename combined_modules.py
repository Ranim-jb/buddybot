# Combined modules from calorie_tracker_combined.py, habit_tracker.py, progress_tracker.py, emergency_help.py, meal_planner.py, motivational_boosts.py, recipe_suggester.py, workout_suggester.py

import sqlite3
import datetime
import random
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt
import io
import base64

# Calorie Tracker functions
def initialize_calorie_db(db_path="calorie_tracker.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            meal TEXT NOT NULL,
            calories INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_meal(meal, calories, date=None, db_path="calorie_tracker.db", daily_target=2000):
    if calories <= 0:
        return "Please enter a valid calorie amount."
    if date is None:
        date = datetime.date.today().isoformat()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO meals (date, meal, calories) VALUES (?, ?, ?)', (date, meal, calories))
    conn.commit()
    conn.close()
    total_calories = get_total_calories(date, db_path)
    feedback = f"Added '{meal}' with {calories} calories. Total for {date}: {total_calories} calories."
    if total_calories > daily_target:
        feedback += f" Warning: You have exceeded your daily calorie target of {daily_target} calories."
    return feedback

def get_total_calories(date=None, db_path="calorie_tracker.db"):
    if date is None:
        date = datetime.date.today().isoformat()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT SUM(calories) FROM meals WHERE date = ?', (date,))
    result = c.fetchone()
    conn.close()
    return result[0] if result[0] is not None else 0

def get_calorie_summary(date=None, db_path="calorie_tracker.db", daily_target=2000):
    if date is None:
        date = datetime.date.today().isoformat()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT meal, calories FROM meals WHERE date = ?', (date,))
    meals = c.fetchall()
    conn.close()
    if not meals:
        return f"No meals logged yet for {date}."
    summary = f"Meals for {date}:\n"
    for meal, calories in meals:
        summary += f"- {meal}: {calories} calories\n"
    total_calories = get_total_calories(date, db_path)
    summary += f"\nTotal Calories: {total_calories}"
    summary += f"\nDaily Target: {daily_target} calories"
    if total_calories > daily_target:
        summary += "\nWarning: You have exceeded your daily calorie target!"
    return summary

def reset_calorie_tracker(date=None, db_path="calorie_tracker.db"):
    if date is None:
        date = datetime.date.today().isoformat()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DELETE FROM meals WHERE date = ?', (date,))
    conn.commit()
    conn.close()
    return f"Calorie tracker has been reset for {date}."

# Initialize calorie tracker DB on module load
initialize_calorie_db()

# Habit Tracker class
# Removed entire HabitTracker class as per user request

# Progress Tracker class
# Removed entire ProgressTracker class as per user request

# Emergency Help class
class EmergencyHelp:
    def __init__(self):
        self.coping_strategies = [
            "Try deep breathing exercises: inhale slowly for 4 seconds, hold for 7 seconds, exhale for 8 seconds.",
            "Take a short walk to clear your mind.",
            "Practice mindfulness or meditation for a few minutes.",
            "Write down your feelings in a journal.",
            "Reach out to a trusted friend or family member."
        ]
        self.mental_wellness_tips = [
            "Remember, it's okay to have difficult days. Be kind to yourself.",
            "Try to maintain a regular sleep schedule.",
            "Engage in activities you enjoy to boost your mood.",
            "Limit caffeine and sugar intake, especially in the evening.",
            "Consider professional help if feelings persist or worsen."
        ]

    def get_coping_strategies(self):
        return "\\n".join(self.coping_strategies)

    def get_mental_wellness_tips(self):
        return "\\n".join(self.mental_wellness_tips)

# Meal Planner function
def get_meal_plan(goal, diet_preference):
    calorie_targets = {
        "lose": 1500,
        "maintain": 2000,
        "gain": 2500
    }
    target_calories = 2000
    goal_lower = goal.lower()
    for key in calorie_targets:
        if key in goal_lower:
            target_calories = calorie_targets[key]
            break
    macro_ratios = {
        "keto": {"protein": 0.25, "fat": 0.70, "carbs": 0.05},
        "mediterranean": {"protein": 0.30, "fat": 0.35, "carbs": 0.35},
        "none": {"protein": 0.30, "fat": 0.30, "carbs": 0.40}
    }
    ratios = macro_ratios.get(diet_preference.lower(), macro_ratios["none"])
    protein_g = int((target_calories * ratios["protein"]) / 4)
    fat_g = int((target_calories * ratios["fat"]) / 9)
    carbs_g = int((target_calories * ratios["carbs"]) / 4)
    plans = {
        "keto": {
            "breakfast": ["Scrambled eggs with avocado", "Keto pancakes with berries"],
            "lunch": ["Grilled chicken salad with olive oil dressing", "Zucchini noodles with pesto and chicken"],
            "dinner": ["Salmon with asparagus", "Beef stir-fry with broccoli"]
        },
        "mediterranean": {
            "breakfast": ["Greek yogurt with nuts and honey", "Oatmeal with fruits"],
            "lunch": ["Lentil soup with a side of whole wheat bread", "Quinoa salad with chickpeas and vegetables"],
            "dinner": ["Grilled fish with roasted vegetables", "Chicken skewers with a side of couscous"]
        },
        "none": {
            "breakfast": ["Oatmeal with fruits and nuts", "Scrambled eggs with whole wheat toast"],
            "lunch": ["Grilled chicken salad", "Turkey and avocado wrap"],
            "dinner": ["Baked salmon with sweet potato", "Lean beef with mixed vegetables"]
        }
    }
    plan = plans.get(diet_preference.lower(), plans["none"])
    meal_plan = {
        "breakfast": random.choice(plan["breakfast"]),
        "lunch": random.choice(plan["lunch"]),
        "dinner": random.choice(plan["dinner"]),
        "macronutrients": {
            "calories": target_calories,
            "protein_g": protein_g,
            "fat_g": fat_g,
            "carbs_g": carbs_g
        }
    }
    return meal_plan

# Motivational Boosts class
class MotivationalBoosts:
    def __init__(self):
        self.quotes = [
            "Keep going, you're doing great!",
            "Every step is progress, no matter how small.",
            "Believe in yourself and all that you are.",
            "You are stronger than you think.",
            "Stay positive, work hard, make it happen.",
            "Consistency is key to success.",
            "Celebrate every small victory 🎉"
        ]
        self.last_quote_date = None

    def get_daily_quote(self):
        today = datetime.date.today()
        if self.last_quote_date == today:
            return None
        self.last_quote_date = today
        return random.choice(self.quotes)

    def get_reminder(self):
        reminders = [
            "Don't forget to stay hydrated! Drink some water 💧",
            "Time to move around! A little stretch goes a long way.",
            "Keep up the great work, you're making progress!",
            "Remember to take deep breaths and relax.",
            "Stay focused and keep pushing towards your goals."
        ]
        return random.choice(reminders)

    def get_praise(self, message="You crushed your calorie goal today! 👏"):
        return message

# Recipe Suggester class
class RecipeSuggester:
    def __init__(self):
        self.recipes = [
            {
                "name": "Keto Avocado Salad",
                "ingredients": ["Avocado", "Lettuce", "Olive oil", "Lemon juice", "Salt"],
                "calories": 350,
                "diet_type": "keto",
                "instructions": "Mix all ingredients and serve chilled."
            },
            {
                "name": "Mediterranean Quinoa Salad",
                "ingredients": ["Quinoa", "Tomatoes", "Cucumber", "Feta cheese", "Olive oil"],
                "calories": 400,
                "diet_type": "mediterranean",
                "instructions": "Cook quinoa, chop vegetables, mix all with olive oil and feta."
            },
            {
                "name": "Grilled Chicken Wrap",
                "ingredients": ["Chicken breast", "Whole wheat wrap", "Lettuce", "Tomato", "Mustard"],
                "calories": 450,
                "diet_type": "none",
                "instructions": "Grill chicken, assemble wrap with vegetables and mustard."
            }
        ]

    def suggest_recipes(self, max_calories=None, diet_type=None):
        filtered = self.recipes
        if diet_type:
            filtered = [r for r in filtered if r["diet_type"] == diet_type.lower()]
        if max_calories:
            filtered = [r for r in filtered if r["calories"] <= max_calories]
        if not filtered:
            return "No recipes found matching your criteria."
        return random.choice(filtered)

# Workout Suggester function
def get_workout_suggestion(fitness_level, time_available, location):
    suggestions = {
        "beginner": {
            "home": {
                "15": ["10 min light cardio (jumping jacks, high knees)", "5 min stretching"],
                "30": ["15 min bodyweight circuit (squats, push-ups, lunges)", "10 min cardio", "5 min stretching"],
                "60": ["20 min bodyweight strength training", "30 min brisk walking or jogging", "10 min stretching"]
            },
            "gym": {
                "30": ["10 min treadmill warm-up", "15 min machine circuit (leg press, chest press)", "5 min cool-down"],
                "60": ["10 min warm-up", "30 min full-body workout using machines", "15 min elliptical", "5 min stretching"]
            }
        },
        "intermediate": {
            "home": {
                "30": ["20 min HIIT workout", "10 min core exercises"],
                "60": ["30 min dumbbell workout", "20 min running", "10 min stretching"]
            },
            "gym": {
                "45": ["10 min warm-up", "30 min free weights (squats, deadlifts, bench press)", "5 min cool-down"],
                "75": ["15 min warm-up", "45 min strength training (split routine)", "15 min rowing machine"]
            }
        }
    }
    workout_plan = suggestions.get(fitness_level.lower(), suggestions["beginner"])
    location_plan = workout_plan.get(location.lower(), workout_plan["home"])
    time_plan = location_plan.get(str(time_available), location_plan[next(iter(location_plan))])
    return "\\n".join(time_plan)
