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

![Screenshot 2025-06-24 091048](https://github.com/user-attachments/assets/e4753444-ea0f-45c1-bcc1-b3b8518b0524)

![Screenshot 2025-06-24 091115](https://github.com/user-attachments/assets/9c2793d5-3af5-4b28-816a-fc594d349e9d)

![Screenshot 2025-06-24 091127](https://github.com/user-attachments/assets/34ac5914-8c2d-42a5-96b2-4eeae09ce9b9)

![Screenshot 2025-06-24 091212](https://github.com/user-attachments/assets/c6a27c58-a3ef-4978-a5f6-f545c1713281)

![Screenshot 2025-06-24 091234](https://github.com/user-attachments/assets/dd9146ae-28df-4fc8-8b8b-b756c4053059)

![Screenshot 2025-06-24 091252](https://github.com/user-attachments/assets/cf6da632-1873-4a24-84a3-a1b654b8cde2)

![Screenshot 2025-06-24 091308](https://github.com/user-attachments/assets/132e998b-9585-4a2c-b5f8-e8472fea230c)

![Screenshot 2025-06-24 091317](https://github.com/user-attachments/assets/a66a4b00-17d0-4837-922a-c97c991beac7)

![Screenshot 2025-06-24 091339](https://github.com/user-attachments/assets/ab1b78f8-932e-4d01-b063-dd9fd8ed3883)

![Screenshot 2025-06-24 091356](https://github.com/user-attachments/assets/e357b95f-45af-4442-b131-836cadc22768)

![Screenshot 2025-06-24 091540](https://github.com/user-attachments/assets/2ef685d1-ce55-464f-9ee9-2bac2669187d)

![Screenshot 2025-06-24 091605](https://github.com/user-attachments/assets/56ce8af4-5400-4569-ba57-aac2f65b8a5a)

![Screenshot 2025-06-24 091625](https://github.com/user-attachments/assets/e6726ddb-8425-4138-85f3-6b3f263bb476)

(You can paste screenshots manually in the GitHub README or attach in the repo.)

✨ Future Enhancements
Add user login and favorite book tracking

Deploy on cloud platforms (e.g. Streamlit Community Cloud, Heroku)

Introduce sentiment analysis of reviews

Add collaborative filtering for enhanced accuracy

📄 License
This project is open-source and available under the MIT License.



