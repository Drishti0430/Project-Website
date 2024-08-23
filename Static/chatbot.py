from flask import Flask, render_template, request , jsonify

import google.generativeai as genai


app = Flask(__name__)
app.secret_key = 'your_secret_key'


API_KEY = 'AIzaSyCnHiPnc81WluNjSklL6lLR5FO_NbHRCfM'
#'AIzaSyCCrYnLhDIgToWeG4u_nPpQcB9uNJMze0U'
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

medical_keywords = [ "Movie Recommendation",
    "Personalized Suggestions",
    "Genre Preferences",
    "Recent Releases",
    "Classic Films",
    "Family-Friendly Movies",
    "Comedy Movies",
    "Romantic Films",
    "Adventure Movies",
    "Documentary Recommendations",
    "Sci-Fi Movies",
    "Horror Films",
    "Animated Movies",
    "Biographical Films",
    "Foreign Films",
    "Book Adaptations",
    "Inspirational Movies",
    "Mystery Films",
    "Action Movies",
    "Romantic Comedy",
    "Mind-Bending Movies",
    "Underrated Films",
    "Cult Classics",
    "Strong Female Leads",
    "Twist Endings",
    "Musicals",
    "Teen Movies",
    "Stunning Visuals",
    "Holiday Movies",
    "Powerful Performances",
    "Historical Dramas",
    "Heist Movies",
    "Feel-Good Films",
    "Crime Dramas",
    "Time Travel Movies",
    "Social Message Films",
    "Trending Movies",
    "Critically Acclaimed",
    "User Preferences",
    "AI-Powered Recommendations",
    "Feedback Loop",
    "Watchlist",
    "Movie Rating",
    "User Behavior Analysis",
    "Sentiment Analysis",
    "Interactive Chatbot",
    "Engagement Metrics",
    "Content Discovery",
    "Niche Films",
    "Recommendation Engine"
    
]




@app.route('/ask', methods=['POST'])
def ask():
    user_message = str(request.form['messageText'])
    
    if not is_medical_query(user_message):
        bot_response_text = "I'm sorry, I can only answer medical-related questions. Please ask a question related to medical topics."
    else:
        bot_response = chat.send_message(user_message)
        bot_response_text = bot_response.text
    
    return jsonify({'status': 'OK', 'answer': bot_response_text})

def is_medical_query(query):
    return any(keyword in query.lower() for keyword in medical_keywords)







