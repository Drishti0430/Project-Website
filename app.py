from flask import Flask, render_template, request , jsonify
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
import pandas as pd

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


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

csv_path = 'datasetFile.csv'  
data = pd.read_csv(csv_path)

X = data[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6']]
y = data['Parameter 9']
X = X.to_numpy()

model1 = LinearRegression()
model1.fit(X, y)
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/main')
def main():
    return render_template('main.html')



@app.route('/')
def home():
    return render_template('Homepage.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        inputs = [float(request.form[field]) for field in ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6']]
        prediction = model1.predict([inputs])
        output = "Argument 1" if prediction[0] >= 0.5 else "Argument 2"
        return render_template('index1.html', prediction_text=f'{output}')
    return render_template('index1.html')


@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')

@app.route('/coding')
def coding():
    return render_template('coding.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = str(request.form['messageText'])
    
    if not is_medical_query(user_message):
        bot_response_text = "I'm sorry, I can only answer movie-related questions. Please ask a question related to medical topics."
    else:
        bot_response = chat.send_message(user_message)
        bot_response_text = bot_response.text
    
    return jsonify({'status': 'OK', 'answer': bot_response_text})

def is_medical_query(query):
    return any(keyword.lower() in query.lower() for keyword in medical_keywords)
# Route to display the .py file

@app.route('/chatbotScript')
def chatbotScript():
    with open('Static/chatbot.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Chatbot Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Highlighted Python Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content

@app.route('/statistical')
def statistical():
    with open('Static/statistical.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Recommendation System</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Highlighted Python Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content
