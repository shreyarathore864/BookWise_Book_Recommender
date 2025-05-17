# BookWise

📚 BookWise – Intelligent Book Recommendation System
BookWise is a smart, personalized book recommendation web application that helps users discover their next favorite read. It analyzes book metadata from Goodreads and Kindle/Amazon datasets and recommends similar books using TF-IDF vectorization and Nearest Neighbors algorithms. The application is built with Python and powered by Streamlit for an interactive user interface.

#LIVE APP--> https://bookwise1.streamlit.app/

🚀 Features
🔍 Search by Title: Suggests books similar to a searched title using semantic similarity.

🎨 Filter by Genre: Browse books categorized under specific genres with an infinite scroll-like "View More" option.

🤖 ML-Based Recommendations: Uses natural language processing and machine learning to find similar books.

📊 Combined Data: Uses metadata from both Goodreads and Kindle to broaden suggestions.

🖼️ Clean UI: Streamlit-powered responsive and styled interface.

📁 Project Structure

BOOK-WISE/
│
├── data/
│   ├── books.csv                  # Goodreads dataset
│   └── kindle_data-v2.csv         # Kindle/Amazon dataset
│
├── main.py                        # Streamlit frontend logic
├── Recommendation.py             # Recommendation logic
├── requirements.txt              # Dependencies
└── README.md                     # Project description

🛠️ Installation & Setup
Clone the repository

git clone https://github.com/your-username/BOOK-WISE.git
cd BOOK-WISE
Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate  # Windows
Install dependencies


pip install -r requirements.txt
Run the application


streamlit run main.py
The app will launch in your browser at http://localhost:8501.

🧠 Tech Stack
Frontend: Streamlit

Backend / Logic: Python, pandas, scikit-learn

Machine Learning: TF-IDF Vectorizer, Nearest Neighbors

Data: Goodreads and Amazon Kindle book datasets (CSV)

📷 Screenshots
Include screenshots of:

Landing Page

Genre-Based Recommendations

Title Search & Similar Books

(You can paste screenshots manually in the GitHub README or attach in the repo.)

✨ Future Enhancements
Add user login and favorite book tracking

Deploy on cloud platforms (e.g. Streamlit Community Cloud, Heroku)

Introduce sentiment analysis of reviews

Add collaborative filtering for enhanced accuracy

📄 License
This project is open-source and available under the MIT License.



