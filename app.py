from flask import Flask, render_template, request, session
from enhanced_craps import EnhancedCrapsAnalyzer, BettingStrategy
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def get_roll_stats(rolls):
    if not rolls:
        return {
            'sevens': 0, 'seven_percent': 0,
            'points': 0, 'point_percent': 0,
            'craps': 0, 'craps_percent': 0,
            'most_common': (0, 0),
            'longest_streak': (0, 0)
        }

    total = len(rolls)
    counts = Counter(rolls)

    # Calculate basic statistics
    sevens = counts[7]
    points = sum(counts[n] for n in [4, 5, 6, 8, 9, 10])
    craps = sum(counts[n] for n in [2, 3, 12])

    # Find most common number
    most_common = counts.most_common(1)[0] if counts else (0, 0)

    # Find longest streak (now checking from newest to oldest)
    longest_streak = (0, 0)  # (number, streak_length)
    if rolls:
        current_streak = (rolls[0], 1)
        for i in range(1, len(rolls)):
            if rolls[i] == current_streak[0]:
                current_streak = (current_streak[0], current_streak[1] + 1)
                if current_streak[1] > longest_streak[1]:
                    longest_streak = current_streak
            else:
                current_streak = (rolls[i], 1)

    return {
        'sevens': sevens,
        'seven_percent': (sevens/total)*100 if total > 0 else 0,
        'points': points,
        'point_percent': (points/total)*100 if total > 0 else 0,
        'craps': craps,
        'craps_percent': (craps/total)*100 if total > 0 else 0,
        'most_common': most_common,
        'longest_streak': longest_streak
    }

def init_session():
    if 'rolls' not in session:
        session['rolls'] = []
    if 'bankroll' not in session:
        session['bankroll'] = 1000
    if 'strategy' not in session:
        session['strategy'] = 'MEDIUM'
    if 'button' not in session:
        session['button'] = 'OFF'

def ensure_minimum_bet(recommendations, min_bet=1):
    """Ensure all bet amounts are at least the minimum bet"""
    for rec in recommendations:
        if rec['amount'] < min_bet:
            rec['amount'] = min_bet
    return recommendations

def process_recommendations(result):
    """Process and validate recommendations"""
    if not result or 'recommendations' not in result:
        return result

    recommendations = result['recommendations']

    # Ensure we have the pass line bet when button is OFF
    if result['button_info']['on_come_out']:
        has_pass_line = any(rec['bet'] == 'Pass Line' for rec in recommendations)
        if not has_pass_line:
            recommendations.append({
                'bet': 'Pass Line',
                'amount': max(round(result['bankroll_info']['base_unit']), 1),
                'reason': 'Come out roll - standard bet',
                'confidence': 1.0,
                'rules': [
                    'Wins on natural (7, 11)',
                    'Loses on craps (2, 3, 12)',
                    'Point number establishes the point'
                ]
            })

    # Ensure minimum bet amounts
    recommendations = ensure_minimum_bet(recommendations)

    # Remove any remaining zero bets
    recommendations = [rec for rec in recommendations if rec['amount'] > 0]

    result['recommendations'] = recommendations
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    init_session()

    if request.method == 'POST':
        if 'reset' in request.form:
            # Update settings without clearing rolls
            session['bankroll'] = float(request.form.get('new_bankroll', 1000))
            session['strategy'] = request.form.get('new_strategy', 'MEDIUM')

        elif 'clear_rolls' in request.form:
            session['rolls'] = []
            session['button'] = 'OFF'

        elif 'roll_history' in request.form:
            # Process roll history input
            history_text = request.form.get('roll_history', '').strip()
            if history_text:
                try:
                    # Split by commas and clean up
                    new_rolls = [int(r.strip()) for r in history_text.split(',')]
                    # Validate each roll
                    valid_rolls = [r for r in new_rolls if 2 <= r <= 12]
                    # Insert in reverse order (newest first)
                    session['rolls'] = valid_rolls[::-1] + session['rolls']
                    # Keep only last 50 rolls
                    session['rolls'] = session['rolls'][:50]
                except ValueError:
                    # Handle invalid input silently
                    pass

        else:
            new_roll = request.form.get('roll')
            new_button = request.form.get('button', 'OFF')

            if new_roll and new_roll.isdigit():
                roll = int(new_roll)
                if 2 <= roll <= 12:
                    session['rolls'].insert(0, roll)
                    session['rolls'] = session['rolls'][:50]

            if new_button in ['OFF'] + [str(x) for x in [4, 5, 6, 8, 9, 10]]:
                session['button'] = new_button

        session.modified = True

    analyzer = EnhancedCrapsAnalyzer(
        bankroll=session['bankroll'],
        strategy_type=BettingStrategy[session['strategy']]
    )

    current_button = None
    if session['button'] != 'OFF':
        try:
            current_button = int(session['button'])
        except (ValueError, TypeError):
            current_button = None

    result = None
    if session['rolls']:
        result = analyzer.recommend_bets(session['rolls'], current_button)
        result = process_recommendations(result)

    roll_stats = get_roll_stats(session['rolls'])

    return render_template(
        'index.html',
        rolls=session['rolls'],
        result=result,
        bankroll=session['bankroll'],
        strategy=session['strategy'],
        roll_stats=roll_stats,
        current_button=session['button']
    )

if __name__ == '__main__':
    app.run(debug=True)
