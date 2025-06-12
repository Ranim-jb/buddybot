# %%
# Import necessary modules and libraries
import os
from dotenv import load_dotenv
# Import specific loaders for document loading
from langchain_community.document_loaders import DirectoryLoader, TextLoader # Keep TextLoader for .txt
# from langchain.document_loaders import PyPDFLoader # Example for PDF
# from langchain.document_loaders import UnstructuredWordDocumentLoader # Example for .docx
from langchain_unstructured import UnstructuredLoader # More general loader (requires 'unstructured')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr
import shutil # Import shutil for file operations
# Import mimetypes to help identify file type
import mimetypes

# Import various modules for different functionalities
from combined_modules import (
    get_meal_plan,
    add_meal,
    get_calorie_summary,
    reset_calorie_tracker,
    get_workout_suggestion,
    MotivationalBoosts,
    RecipeSuggester,
    EmergencyHelp
)


# Load environment variables from .env file
load_dotenv()

class BuddyBot:
    def __init__(self, knowledge_dir="knowledge_base"):
        """
        Initialize BuddyBot with a knowledge directory.

        Args:
            knowledge_dir (str): Path to directory containing text files for the knowledge base
        """
        self.knowledge_dir = knowledge_dir
        # Initialize the language model with specified parameters
        self.llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
        self.vectorstore = None  # Vector store for document embeddings
        self.qa_chain = None  # Retrieval QA chain for answering questions
        self.history = []  # Chat history
        self._initialize_bot()  # Initialize the bot components

    def _initialize_bot(self):
        """Initialize the knowledge base and QA chain."""
        # Load documents from the knowledge directory
        if os.path.exists(self.knowledge_dir):
            # Use DirectoryLoader with UnstructuredLoader to load various file types
            # UnstructuredLoader requires the 'unstructured' library and dependencies
            try:
                loader = DirectoryLoader(
                    self.knowledge_dir,
                    loader_cls=UnstructuredLoader,  # Use UnstructuredLoader for broad file support
                    show_progress=True,
                    recursive=True,  # Recursively load files in subdirectories
                    use_multithreading=True  # Speed up loading with multithreading
                )
                documents = loader.load()
                # Fallback to TextLoader for any files UnstructuredLoader might miss or fail on
                # This fallback is a simple approach and might be refined later
            except Exception as e:
                print(f"Error loading documents using UnstructuredLoader: {e}")
                print("Attempting to load only .txt files using TextLoader as a fallback.")
                # Load only .txt files if UnstructuredLoader fails or is not installed
                text_files = [os.path.join(self.knowledge_dir, f) for f in os.listdir(self.knowledge_dir) if f.endswith('.txt')]
                documents = []
                for file_path in text_files:
                    try:
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
                    except Exception as e_txt:
                        print(f"Error loading text file {file_path}: {e_txt}")

        else:
            # Create the knowledge directory if it does not exist
            os.makedirs(self.knowledge_dir, exist_ok=True)
            documents = []
            print(f"Created empty knowledge directory at {self.knowledge_dir}")  # Log creation

        # Split documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Check if there are any splits before creating vector store and QA chain
        if not splits:
            print("Knowledge base is empty or no text could be extracted from documents. Add valid documents to enable chat functionality.")
            self.vectorstore = None
            self.qa_chain = None
            return  # Exit if no documents found

        # Create vector store from document chunks using embeddings
        embeddings = HuggingFaceEmbeddings()
        try:
            self.vectorstore = FAISS.from_documents(splits, embeddings)
        except Exception as e:
            print(f"Error creating vector store from splits: {e}")
            print("Vector store creation failed. Bot will not be able to answer questions from the knowledge base.")
            self.vectorstore = None
            self.qa_chain = None
            return

        # Define prompt template for the retrieval QA chain
        prompt_template = """You are WeightLossBuddy, an AI assistant with a persona of a supportive and knowledgeable coach who specializes in weight loss topics. Your primary goal is to provide clear, encouraging, and science-backed information based *only* on the provided context.

        **Persona Guidelines:**
        - **Tone**: Always be positive, motivating, and practical.
        - **Role**: You are an informational coach, not a doctor. You provide information, not medical advice.
        - **Safety**: If a user expresses concerns about eating disorders or extreme dieting, gently guide them towards consulting a healthcare professional.

        **Response Instructions:**
        1.  **Analyze the Question**: Understand the user's intent. Are they asking for information, tips, or motivation?
        2.  **Consult the Context**: Base your answer strictly
        3.  **Structure Your Answer**:
            - If the answer involves a list (e.g., food items, exercises), use bullet points.
            - Keep paragraphs concise and actionable.
        4.  **Handle "I Don't Know" Scenarios**: If the context does not contain the answer, state clearly that you don't have information on that specific topic. For example: "I don't have information on that in my knowledge base. For personal health questions, it's always best to consult with a healthcare professional or a registered dietitian."
        5.  **Negative Constraints**:
            - **DO NOT** provide medical advice, diagnoses, or personalized meal or workout plans.
            - **DO NOT** make up information that is not in the context.
            - **DO NOT** promote unhealthy or extreme weight loss methods.

        **Context:**
        {context}

        **Question:**
        {question}

        **Helpful Answer:**"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        print(f"Initialized BuddyBot with {len(documents)} documents ({len(splits)} chunks).") # Print to console/logs


    def chat(self, question, history):
        """
        Chat with BuddyBot by asking a question.

        Args:
            question (str): The question to ask BuddyBot
            history (list): The conversation history

        Returns:
            list: Updated conversation history
        """
        if not self.qa_chain:
            history.append((question, "I'm not properly initialized or the knowledge base is empty. Add documents using the 'Add Document' tab to enable chat."))
            return history

        try:
            result = self.qa_chain({"query": question})
            answer = result["result"]
            sources = [os.path.basename(doc.metadata.get("source", "N/A")) for doc in result.get("source_documents", [])]

            # Format the output for the Gradio interface
            if sources:
                response = f"{answer}\n\nSources: {', '.join(sources)}"
            else:
                response = answer
            
            history.append((question, response))
            return history

        except Exception as e:
            print(f"An error occurred during chat: {e}") # Print to console/logs
            history.append((question, "Sorry, I encountered an error while trying to answer that."))
            return history


    def add_document(self, file_obj):
        """
        Add a new document to the knowledge base from a Gradio File object.

        Args:
            file_obj (gr.File): Gradio File object uploaded by the user.

        Returns:
            str: Status message
        """
        if file_obj is None:
            return "No file uploaded."

        # Gradio provides the path to the temporary file
        file_path = file_obj.name
        original_file_name = os.path.basename(file_path)

        if not os.path.exists(file_path):
            return f"Error: Uploaded file not found at temporary path {file_path}"

        # Copy the document to the knowledge base directory
        try:
            dest_file_name = original_file_name
             # Ensure the filename is unique if it already exists
            base, ext = os.path.splitext(dest_file_name)
            dest_file_path = os.path.join(self.knowledge_dir, dest_file_name)
            counter = 1
            while os.path.exists(dest_file_path):
                dest_file_name = f"{base}_{counter}{ext}"
                dest_file_path = os.path.join(self.knowledge_dir, dest_file_name)
                counter += 1


            shutil.copy(file_path, dest_file_path)
            status_message = f"Copied temporary file to knowledge base: {os.path.basename(dest_file_path)}\n"
            print(status_message.strip()) # Print to console/logs

        except Exception as e:
            print(f"Error copying file {original_file_name}: {e}") # Print to console/logs
            return f"Error copying file '{original_file_name}' to knowledge base: {e}"


        # Load the new document using an appropriate loader
        try:
            # Determine loader based on file extension or mime type (basic)
            mime_type, _ = mimetypes.guess_type(dest_file_path)
            print(f"Attempting to load file: {os.path.basename(dest_file_path)} with detected MIME type: {mime_type}")

            loader = UnstructuredLoader(dest_file_path)
            # You could add more specific loaders here and try them based on mime_type/extension
            # For example:
            # if mime_type == 'application/pdf':
            #     loader = PyPDFLoader(dest_file_path)
            # elif ext.lower() == '.docx':
            #     loader = UnstructuredWordDocumentLoader(dest_file_path)
            # else:
            #     loader = TextLoader(dest_file_path) # Default to TextLoader or UnstructuredFileLoader


            new_doc = loader.load()

        except Exception as e:
            print(f"Error loading copied file {dest_file_path}: {e}") # Print to console/logs
            # Clean up the copied file if loading fails
            if os.path.exists(dest_file_path):
                 os.remove(dest_file_path)
                 print(f"Removed failed document: {os.path.basename(dest_file_path)}")
            return f"Error loading copied file '{os.path.basename(dest_file_path)}': {e}\nThis file type might not be supported or requires additional dependencies (e.g., `unstructured`, `pandoc`, etc.)."


        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        new_splits = text_splitter.split_documents(new_doc)

        if not new_splits:
            print(f"No text found in {dest_file_path} after splitting. Document not added.") # Print to console/logs
            # Clean up the copied file if splitting fails (e.g., empty PDF)
            if os.path.exists(dest_file_path):
                 os.remove(dest_file_path)
                 print(f"Removed empty document: {os.path.basename(dest_file_path)}")
            return f"No text could be extracted from '{os.path.basename(dest_file_path)}'. Document not added."

        # Add to vector store
        embeddings = HuggingFaceEmbeddings()
        try:
            if self.vectorstore:
                self.vectorstore.add_documents(new_splits)
                status_message += f"Successfully extracted {len(new_splits)} chunks and added '{os.path.basename(dest_file_path)}' to vector store."
                print(f"Successfully added {os.path.basename(dest_file_path)} to vector store.") # Print to console/logs
            else:
                # If vectorstore was None (empty knowledge base), create a new one
                self.vectorstore = FAISS.from_documents(new_splits, embeddings)
                status_message += f"Created new vector store with {len(new_splits)} chunks from '{os.path.basename(dest_file_path)}'."
                print(f"Created new vector store with {os.path.basename(dest_file_path)}.") # Print to console/logs

        except Exception as e:
             print(f"Error adding documents to vector store: {e}") # Print to console/logs
             # Clean up the copied file if adding to vector store fails
             if os.path.exists(dest_file_path):
                 os.remove(dest_file_path)
                 print(f"Removed failed document: {os.path.basename(dest_file_path)}")
             self.vectorstore = None # Ensure vectorstore is None on failure
             self.qa_chain = None # Ensure qa_chain is None on failure
             return f"Error adding '{os.path.basename(dest_file_path)}' to vector store: {e}\nBot reset required."


        # Reinitialize QA chain to use the updated vectorstore
        # IMPORTANT: Re-initializing the entire bot might reset other things.
        # A better approach might be to just update the retriever if the vectorstore is modified.
        # For simplicity here, we re-initialize as it was done before.
        self._initialize_bot()

        # Check if _initialize_bot was successful in creating/updating the QA chain
        if self.qa_chain:
             status_message += f" Bot re-initialized successfully."
             print("Bot re-initialized successfully.") # Print to console/logs
        else:
             status_message += f" Bot re-initialization failed. Check console output for details."
             print("Bot re-initialization failed.") # Print to console/logs


        return status_message

    def list_documents(self):
        """
        List the documents currently in the knowledge base directory.

        Returns:
            list: List of document filenames
        """
        if not os.path.exists(self.knowledge_dir):
            return []
        # List all files in the directory (since we now accept more than just .txt)
        return [f for f in os.listdir(self.knowledge_dir) if os.path.isfile(os.path.join(self.knowledge_dir, f))]

    def clear_documents(self):
        """
        Remove all documents from the knowledge base directory and reset the bot.

        Returns:
            str: Status message
        """
        if os.path.exists(self.knowledge_dir):
            try:
                # Get list of files first to print later (all files now)
                files_to_remove = [f for f in os.listdir(self.knowledge_dir) if os.path.isfile(os.path.join(self.knowledge_dir, f))]
                shutil.rmtree(self.knowledge_dir)
                status_message = f"Removed knowledge directory: {self.knowledge_dir}\n"
                print(status_message.strip()) # Print to console/logs
                if files_to_remove:
                     status_message += "Removed files:\n"
                     for f in files_to_remove:
                         status_message += f"- {f}\n"
                         print(f"- {f}") # Print to console/logs
            except Exception as e:
                print(f"Error clearing knowledge directory: {e}") # Print to console/logs
                return f"Error clearing knowledge directory: {e}"
        else:
            status_message = f"Knowledge directory {self.knowledge_dir} does not exist.\n"
            print(status_message.strip()) # Print to console/logs


        # Reinitialize the bot with an empty knowledge base
        self._initialize_bot()
        status_message += "Knowledge base cleared and bot reset."
        print("Knowledge base cleared and bot reset.") # Print to console/logs
        return status_message


# Instantiate the bot globally or pass it to interface functions
buddy_bot = BuddyBot()
calorie_tracker = None  # Deprecated, use combined_modules functions
motivational_boosts = MotivationalBoosts()
recipe_suggester = RecipeSuggester()
emergency_help = EmergencyHelp()

# Gradio interface functions
def chat_interface(question, history):
    """Function to handle chat input from Gradio."""
    history = history or []
    return buddy_bot.chat(question, history)

def add_document_interface(file_obj):
    """Function to handle file upload for adding documents."""
    return buddy_bot.add_document(file_obj)

def list_documents_interface():
    """Function to list documents for the interface."""
    docs = buddy_bot.list_documents()
    if docs:
        return "Documents:\n" + "\n".join([f"- {doc}" for doc in docs])
    else:
        return "The knowledge base is currently empty."

def clear_documents_interface():
    """Function to clear documents via the interface."""
    return buddy_bot.clear_documents()

def calculate_bmi(height_cm, weight_kg):
    """Calculates BMI from height in cm and weight in kg."""
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100  # Convert height from cm to meters
        bmi = weight_kg / (height_m ** 2)  # Calculate BMI using formula
        category = ""
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obesity"
        return f"Your BMI is {bmi:.2f} ({category})"
    return "Please enter valid height and weight."


# Create Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
    css="""
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #e0ffe7 0%, #e9f0f7 100%);
        margin: 0;
        padding: 0;
    }
    .gradio-container {
        max-width: 1150px;
        margin: 30px auto;
        padding: 38px;
        background: #ffffffee;
        border-radius: 22px;
        box-shadow: 0 10px 32px rgba(0,0,0,0.14);
        border: 2px solid #b2f2bb;
        position: relative;
        overflow: hidden;
    }
    .gradio-container:before {
        content: "🌈";
        position: absolute;
        top: -30px;
        right: 30px;
        font-size: 60px;
        opacity: 0.18;
        pointer-events: none;
    }
    h1, h2, h3 {
        color: #2d3e50;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .gr-button {
        background: linear-gradient(90deg, #27ae60 0%, #2196f3 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 18px;
        transition: background 0.2s, transform 0.1s;
        box-shadow: 0 3px 12px rgba(39,174,96,0.12);
        cursor: pointer;
    }
    .gr-button:hover {
        background: linear-gradient(90deg, #2196f3 0%, #27ae60 100%);
        transform: scale(1.04) rotate(-2deg);
    }
    /* Change white buttons (variant=stop) to a custom color */
    .gr-button[aria-label][data-testid="stop-button"], .gr-button.variant-stop, .gr-button:has(svg[data-testid="StopIcon"]) {
        background: linear-gradient(90deg, #ffb347 0%, #ffcc33 100%) !important;
        color: #2d3e50 !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(255, 204, 51, 0.12) !important;
    }
    .gr-button[aria-label][data-testid="stop-button"]:hover, .gr-button.variant-stop:hover, .gr-button:has(svg[data-testid="StopIcon"]):hover {
        background: linear-gradient(90deg, #ffcc33 0%, #ffb347 100%) !important;
        color: #2d3e50 !important;
    }
    .gr-textbox, .gr-dropdown, .gr-number, .gr-textarea {
        border: 1.5px solid #b2bec3;
        border-radius: 11px;
        padding: 13px;
        font-size: 16px;
        color: #2c3e50;
        background: #f8fafc;
        transition: border 0.2s;
    }
    .gr-textbox:focus, .gr-dropdown:focus, .gr-number:focus, .gr-textarea:focus {
        border: 1.5px solid #27ae60;
        outline: none;
    }
    .gr-chatbot {
        border-radius: 18px;
        border: 2px solid #b2f2bb;
        background-color: #f7f9fa;
        padding: 22px;
        box-shadow: 0 2px 8px rgba(39,174,96,0.06);
    }
    .gr-tabs {
        margin-top: 32px;
    }
    .section-header {
        margin-bottom: 19px;
        font-size: 23px;
        color: #27ae60;
        font-weight: 800;
        border-bottom: 2.5px dashed #27ae60;
        padding-bottom: 9px;
        letter-spacing: 0.7px;
        background: linear-gradient(90deg, #e0ffe7 0%, #e9f0f7 100%);
        border-radius: 8px;
    }
    .input-row {
        margin-bottom: 24px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #7f8c8d;
        font-size: 16px;
        letter-spacing: 0.2px;
    }
    .emoji-title {
        font-size: 42px;
        margin-right: 10px;
        vertical-align: middle;
    }
    .gradio-container:after {
        content: "✨";
        position: absolute;
        bottom: -30px;
        left: 30px;
        font-size: 60px;
        opacity: 0.13;
        pointer-events: none;
    }
    """
) as demo:
    gr.Markdown(
        "<h1 style='text-align:center;'><span class='emoji-title'>🌱</span> <span style='color:#27ae60;'>Weight Loss Buddy</span> <span class='emoji-title'>🎉</span></h1>"
        "<p style='text-align:center; color:#219150; font-size:20px;'>Your joyful AI companion for healthy weight loss, motivation, and progress tracking!<br>Let's make your journey fun and positive! <span style='font-size:22px;'>💪😊</span></p>",
        elem_id="title"
    )

    with gr.Tabs():
        # Chat Tab
        with gr.Tab("💬 Chat"):
            gr.Markdown("### 💬 Chat with WeightLossBuddy <span style='font-size:22px;'>🤗</span>", elem_classes="section-header")
            chatbot = gr.Chatbot(label="WeightLossBuddy", avatar_images=("🤖", "🧑"))
            msg = gr.Textbox(label="Your Question", placeholder="Ask me about weight loss, nutrition, or motivation... 🌟")
            clear = gr.Button("Clear Conversation 🧹")
            chat_history = gr.State([])

            def user(user_message, history):
                return "", history + [[user_message, None]]

            msg.submit(chat_interface, [msg, chat_history], [chatbot], queue=True).then(
                lambda: gr.update(value=""), None, [msg], queue=False
            )
            clear.click(lambda: [], None, chatbot, queue=True)

        # Add Document Tab
        with gr.Tab("📄 Add Document"):
            gr.Markdown("### 📄 Add Document to Knowledge Base <span style='font-size:20px;'>📚</span>", elem_classes="section-header")
            file_upload = gr.File(label="Upload Document (PDF, DOCX, TXT, etc.)")
            add_status_output = gr.Textbox(label="Status", interactive=False)
            add_doc_button = gr.Button("Add Document 🚀")

            add_doc_button.click(
                add_document_interface,
                inputs=file_upload,
                outputs=add_status_output
            )

        # Manage Knowledge Base Tab
        with gr.Tab("🗂️ Manage Knowledge Base"):
            gr.Markdown("### 🗂️ Manage Knowledge Base Documents <span style='font-size:20px;'>📝</span>", elem_classes="section-header")
            list_docs_button = gr.Button("List Documents 📃")
            list_docs_output = gr.Textbox(label="Documents in KB", interactive=False)

            clear_docs_button = gr.Button("Clear Knowledge Base 🧹", variant="stop")
            clear_status_output = gr.Textbox(label="Clear Status", interactive=False)

            list_docs_button.click(
                list_documents_interface,
                inputs=None,
                outputs=list_docs_output
            )

            clear_docs_button.click(
                clear_documents_interface,
                inputs=None,
                outputs=clear_status_output
            )

        # BMI Calculator Tab
        with gr.Tab("⚖️ BMI Calculator"):
            gr.Markdown("### ⚖️ Calculate Your Body Mass Index (BMI) <span style='font-size:20px;'>📏</span>", elem_classes="section-header")
            with gr.Row():
                height_input = gr.Number(label="Height (cm)")
                weight_input = gr.Number(label="Weight (kg)")
            bmi_button = gr.Button("Calculate BMI 🎯")
            bmi_output = gr.Textbox(label="Your BMI Result", interactive=False)

            bmi_button.click(
                calculate_bmi,
                inputs=[height_input, weight_input],
                outputs=bmi_output
            )

        # Meal Planner Tab
        with gr.Tab("🍽️ Meal Planner"):
            gr.Markdown("### 🍽️ Get a Personalized Meal Plan <span style='font-size:20px;'>🥗</span>", elem_classes="section-header")
            with gr.Row():
                goal_input = gr.Textbox(label="Your Goal (e.g., lose 5kg in 2 months)")
                diet_preference_input = gr.Dropdown(
                    label="Diet Preference",
                    choices=["Keto", "Mediterranean", "None"],
                    value="None"
                )
            meal_plan_button = gr.Button("Get Meal Plan 🥑")
            meal_plan_output = gr.JSON(label="Your Meal Plan")

            meal_plan_button.click(
                get_meal_plan,
                inputs=[goal_input, diet_preference_input],
                outputs=meal_plan_output
            )

        # Calorie Tracker Tab
        with gr.Tab("🔥 Calorie Tracker"):
            gr.Markdown("### 🔥 Track Your Daily Calorie Intake <span style='font-size:20px;'>🍎</span>", elem_classes="section-header")
            with gr.Row(elem_classes="input-row"):
                meal_input = gr.Textbox(label="Meal/Snack")
                calories_input = gr.Number(label="Calories", visible=False)
            with gr.Row():
                add_meal_button = gr.Button("Add Meal 🍽️")
                reset_button = gr.Button("Reset Tracker 🔄", variant="stop")
            calorie_summary_output = gr.Textbox(label="Today's Summary", interactive=False)

            def add_meal_auto_calculate(meal_text):
                # Here you can implement logic to automatically calculate calories based on meal_text
                # For now, we will simulate with a placeholder value or simple logic
                # Example: calories = len(meal_text) * 10 (just a dummy calculation)
                calories = len(meal_text) * 10 if meal_text else 0
                return add_meal(meal_text, calories)


            add_meal_button.click(
                add_meal_auto_calculate,
                inputs=[meal_input],
                outputs=calorie_summary_output
            )
            reset_button.click(
                reset_calorie_tracker,
                inputs=None,
                outputs=calorie_summary_output
            )

        # Workout Suggester Tab
        with gr.Tab("🏋️ Workout Suggester"):
            gr.Markdown("### 🏋️ Get a Workout Suggestion <span style='font-size:20px;'>💪</span>", elem_classes="section-header")
            with gr.Row():
                fitness_level_input = gr.Dropdown(
                    label="Fitness Level",
                    choices=["Beginner", "Intermediate"],
                    value="Beginner"
                )
                time_available_input = gr.Number(label="Time Available (minutes)", value=30)
                location_input = gr.Dropdown(
                    label="Location",
                    choices=["Home", "Gym"],
                    value="Home"
                )
            workout_suggestion_button = gr.Button("Get Workout Suggestion 🏃‍♂️")
            workout_suggestion_output = gr.Textbox(label="Your Workout Suggestion", interactive=False)

            workout_suggestion_button.click(
                get_workout_suggestion,
                inputs=[fitness_level_input, time_available_input, location_input],
                outputs=workout_suggestion_output
            )

        # Motivational Boosts Tab
        with gr.Tab("🌟 Motivational Boosts"):
            gr.Markdown("### 🌟 Daily Motivation and Reminders <span style='font-size:20px;'>✨</span>", elem_classes="section-header")
            daily_quote_button = gr.Button("Get Daily Quote 💡")
            daily_quote_output = gr.Textbox(label="Daily Quote", interactive=False)
            reminder_button = gr.Button("Get Reminder ⏰")
            reminder_output = gr.Textbox(label="Reminder", interactive=False)
            praise_button = gr.Button("Get Praise 🥳")
            praise_output = gr.Textbox(label="Praise", interactive=False)

            daily_quote_button.click(
                lambda: motivational_boosts.get_daily_quote() or "Quote already given today.",
                inputs=None,
                outputs=daily_quote_output
            )
            reminder_button.click(
                motivational_boosts.get_reminder,
                inputs=None,
                outputs=reminder_output
            )
            praise_button.click(
                motivational_boosts.get_praise,
                inputs=None,
                outputs=praise_output
            )

        # Recipe Suggester Tab
        with gr.Tab("🍳 Recipe Suggester"):
            gr.Markdown("### 🍳 Get Recipe Suggestions <span style='font-size:20px;'>🍳</span>", elem_classes="section-header")
            max_calories_input = gr.Number(label="Max Calories")
            diet_type_input = gr.Dropdown(
                label="Diet Type",
                choices=["keto", "mediterranean", "none"],
                value="none"
            )
            suggest_recipe_button = gr.Button("Suggest Recipe 🍳")
            recipe_output = gr.Textbox(label="Recipe Suggestion", interactive=False)

            def suggest_recipe(max_calories, diet_type):
                recipe = recipe_suggester.suggest_recipes(max_calories, diet_type)
                if isinstance(recipe, str):
                    return recipe
                else:
                    ingredients = ", ".join(recipe["ingredients"])
                    instructions = recipe["instructions"]
                    return f"{recipe['name']}:\nIngredients: {ingredients}\nInstructions: {instructions}\nCalories: {recipe['calories']}"

            suggest_recipe_button.click(
                suggest_recipe,
                inputs=[max_calories_input, diet_type_input],
                outputs=recipe_output
            )

        # Emergency Help Tab
        with gr.Tab("🚨 Emergency Help"):
            gr.Markdown("### 🚨 Get Coping Strategies and Mental Wellness Tips <span style='font-size:20px;'>🧘‍♂️</span>", elem_classes="section-header")
            coping_button = gr.Button("Get Coping Strategies 🛟")
            coping_output = gr.Textbox(label="Coping Strategies", interactive=False)
            wellness_button = gr.Button("Get Mental Wellness Tips 🌈")
            wellness_output = gr.Textbox(label="Mental Wellness Tips", interactive=False)

            coping_button.click(
                emergency_help.get_coping_strategies,
                inputs=None,
                outputs=coping_output
            )
            wellness_button.click(
                emergency_help.get_mental_wellness_tips,
                inputs=None,
                outputs=wellness_output
            )

    gr.Markdown(
        "<div class='footer'>"
        "Made with <span style='color:#e25555;'>❤️</span> and <span style='color:#27ae60;'>joy🌟</span>"
        "</div>"
    )



if __name__ == "__main__":
    print("Starting BuddyBot Gradio Interface...")
    # The launch() method creates a public link (share=True) or a local server.
    # You can add server_name="0.0.0.0" to make it accessible on your local network.
    demo.launch()
