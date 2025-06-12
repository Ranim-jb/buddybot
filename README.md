# BuddyBot: Your Weight Loss Assistant

BuddyBot is a conversational AI designed to provide information, support, and personalized assistance for weight loss and healthy living. It uses a knowledge base of documents to answer your questions in a caring and motivating manner.

## Features

* **Conversational AI**: Powered by the Groq API with the Llama3 model for fast and relevant responses.
* **Custom Knowledge Base**: Add your own documents (PDFs, text files, etc.) to the knowledge base, and BuddyBot will use them to answer questions.
* **Personalized Meal Planning**: Get meal plans based on your goals and diet preferences.
* **Calorie Tracking**: Log your meals and snacks, with automatic calorie calculation.
* **Workout Suggestions**: Receive workout routines tailored to your fitness level and available time.
* **Progress Tracking**: Log weight, waist size, steps, and mood, and view progress charts.
* **Motivational Boosts**: Daily motivational quotes, reminders, and praise messages.
* **Habit Tracking**: Track habits with daily check-ins and weekly summaries.
* **Recipe Suggestions**: Find healthy recipes filtered by calorie count and diet type.
* **Emergency Help**: Access coping strategies and mental wellness tips.
* **Persistent Storage**: Uses SQLite and optional Firebase integration for data storage.
* **User-Friendly Interface**: A modern and intuitive interface built with Gradio.

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your environment variables:**
    Create a `.env` file in the root of the project and add your Groq API key:
    ```
    GROQ_API_KEY="your-groq-api-key"
    ```

5. **(Optional) Firebase Setup:**
    If you want to enable Firebase syncing, install the Firebase Admin SDK:
    ```bash
    pip install firebase-admin
    ```
    And configure your Firebase credentials as needed.

## How to Use

1. **Run the application:**
    ```bash
    python buddybot.py
    ```

2. **Open the interface:**
    Open your web browser and go to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

3. **Interact with BuddyBot:**
    * **Chat**: Use the "Chat" tab to ask questions.
    * **Add Documents**: Use the "Add Document" tab to upload new documents to the knowledge base.
    * **Manage Knowledge Base**: Use the "Manage Knowledge Base" tab to list or clear the documents in the knowledge base.
    * **BMI Calculator**: Calculate your Body Mass Index.
    * **Meal Planner**: Get personalized meal plans.
    * **Calorie Tracker**: Log meals and track calories with automatic calculation.
    * **Workout Suggester**: Get workout suggestions.
    * **Progress Tracker**: Log and view your progress with charts.
    * **Motivational Boosts**: Receive daily motivation and reminders.
    * **Habit Tracker**: Track your habits and get weekly summaries.
    * **Recipe Suggester**: Find healthy recipes.
    * **Emergency Help**: Access coping strategies and mental wellness tips.

## License

This project is licensed under the MIT License.
    GROQ_API_KEY="your-groq-api-key"
    pip install -r requirements.txt
