import numpy as np
from collections import Counter, defaultdict
from enum import Enum
from typing import Dict, List, Tuple

class BettingStrategy(Enum):
    CONSERVATIVE = 'conservative'
    MEDIUM = 'medium'
    AGGRESSIVE = 'aggressive'

class MarkovChain:
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.state_counts = defaultdict(int)

    def train(self, sequence: List[int]):
        for i in range(len(sequence) - self.order):
            state = tuple(sequence[i:i + self.order])
            next_value = sequence[i + self.order]
            self.transitions[state][next_value] += 1
            self.state_counts[state] += 1

    def get_probability(self, state: tuple, next_value: int) -> float:
        if state not in self.state_counts or self.state_counts[state] == 0:
            return 0
        return self.transitions[state][next_value] / self.state_counts[state]

class ButtonState:
    OFF = 'OFF'
    ON = 'ON'

    def __init__(self, button_placement=None):
        self.button = button_placement
        self.point_numbers = [4, 5, 6, 8, 9, 10]
        self.current_roll = None

    def set_button(self, number):
        if number in self.point_numbers:
            self.button = number
        elif number is None:
            self.button = None
        else:
            raise ValueError(f"Invalid button placement: {number}")

    def set_current_roll(self, roll):
        self.current_roll = roll

    def get_state(self):
        return {
            'button': self.button if self.button else 'OFF',
            'on_come_out': self.button is None,
            'can_place_bet': self.button is not None,
            'current_roll': self.current_roll
        }

class EnhancedCrapsAnalyzer:
    def __init__(self, bankroll=300, strategy_type=BettingStrategy.MEDIUM):
        self.bankroll = bankroll
        self.strategy_type = strategy_type
        self.markov_chain = MarkovChain(order=2)
        self._init_strategy_parameters()

        # Core game parameters
        self.point_numbers = [4, 5, 6, 8, 9, 10]
        self.naturals = [7, 11]
        self.craps_numbers = [2, 3, 12]

        # Odds and probabilities
        self.probability_map = {
            2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
            7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
        }
        self.place_bet_odds = {
            4: 9/5, 5: 7/5, 6: 7/6, 8: 7/6, 9: 7/5, 10: 9/5
        }
        self.true_odds = {
            4: 2/1, 5: 3/2, 6: 6/5, 8: 6/5, 9: 3/2, 10: 2/1
        }

        # Button state tracking
        self.button_state = ButtonState()
        self.rolling_window_size = 20

    def _init_strategy_parameters(self):
        if self.strategy_type == BettingStrategy.CONSERVATIVE:
            self.base_unit = max(round(self.bankroll * 0.01), 1)
            self.max_place_bets = 1
            self.odds_multiplier = 2
            self.trend_threshold = 1.3
            self.confidence_threshold = 0.3
        elif self.strategy_type == BettingStrategy.MEDIUM:
            self.base_unit = max(round(self.bankroll * 0.02), 1)
            self.max_place_bets = 2
            self.odds_multiplier = 3
            self.trend_threshold = 1.2
            self.confidence_threshold = 0.2
        else:  # AGGRESSIVE
            self.base_unit = max(round(self.bankroll * 0.03), 1)
            self.max_place_bets = 3
            self.odds_multiplier = 5
            self.trend_threshold = 1.1
            self.confidence_threshold = 0.15

    def calculate_kelly_bet(self, win_prob: float, odds: float) -> float:
        """Calculate optimal bet size using Kelly Criterion"""
        q = 1 - win_prob
        if win_prob * (odds + 1) - 1 <= 0:
            return 0
        return (win_prob * (odds + 1) - 1) / odds

    def analyze_patterns(self, rolls):
        return {
            'sequences': self._analyze_sequences(rolls),
            'intervals': self._analyze_intervals(rolls),
            'cycles': self._find_cycles(rolls),
            'conditional_probs': self._analyze_conditional_probs(rolls),
            'streaks': self._analyze_streaks(rolls)
        }

    def _analyze_sequences(self, rolls, max_length=5):
        sequences = defaultdict(list)
        for length in range(2, max_length + 1):
            for i in range(len(rolls) - length):
                sequence = tuple(rolls[i:i+length])
                if i + length < len(rolls):
                    sequences[sequence].append(rolls[i+length])
        return {seq: Counter(next_nums) for seq, next_nums in sequences.items() if len(next_nums) >= 3}

    def _analyze_intervals(self, rolls):
        intervals = defaultdict(list)
        for num in range(2, 13):
            positions = [i for i, r in enumerate(rolls) if r == num]
            for i in range(1, len(positions)):
                intervals[num].append(positions[i] - positions[i-1])
        return {num: Counter(gaps) for num, gaps in intervals.items()}

    def _find_cycles(self, rolls, min_length=3, max_length=20):
        cycles = {}
        for length in range(min_length, max_length + 1):
            for i in range(len(rolls) - length):
                pattern = tuple(rolls[i:i+length])
                count = 0
                for j in range(i + length, len(rolls) - length + 1, length):
                    if tuple(rolls[j:j+length]) == pattern:
                        count += 1
                if count > 1:
                    cycles[pattern] = count
        return cycles

    def _analyze_conditional_probs(self, rolls, lookback=3):
        conditionals = defaultdict(Counter)
        for i in range(len(rolls) - lookback):
            condition = tuple(rolls[i:i+lookback])
            if i + lookback < len(rolls):
                conditionals[condition][rolls[i+lookback]] += 1
        return conditionals

    def _analyze_streaks(self, rolls):
        if not rolls:
            return {}

        streaks = defaultdict(list)
        current_streak = {'number': rolls[0], 'length': 1}

        for i in range(1, len(rolls)):
            if rolls[i] == current_streak['number']:
                current_streak['length'] += 1
            else:
                streaks[current_streak['number']].append(current_streak['length'])
                current_streak = {'number': rolls[i], 'length': 1}

        # Add the last streak
        streaks[current_streak['number']].append(current_streak['length'])

        return {num: {'max': max(lengths), 'avg': sum(lengths)/len(lengths)}
                for num, lengths in streaks.items() if lengths}

    def calculate_pattern_confidence(self, rolls: List[int], target: int) -> float:
        recent_rolls = rolls[-self.rolling_window_size:]
        if len(recent_rolls) < 2:
            return 0.0

        # Markov prediction
        state = tuple(recent_rolls[-2:])
        markov_prob = self.markov_chain.get_probability(state, target)

        # Frequency analysis
        freq = Counter(recent_rolls)
        expected_freq = self.probability_map[target]
        freq_ratio = freq.get(target, 0) / len(recent_rolls) / expected_freq

        # Streak analysis
        streak_boost = 0.0
        streaks = self._analyze_streaks(recent_rolls)
        if target in streaks:
            streak_info = streaks[target]
            streak_boost = min(streak_info['avg'] / 4, 0.2)  # Cap streak boost at 0.2

        # Combine predictors with weights
        confidence = (0.5 * markov_prob +
                     0.3 * min(freq_ratio, 2.0) / 2.0 +
                     0.2 * streak_boost)

        return min(confidence, 1.0)

    def predict_next_roll(self, rolls):
        self.markov_chain.train(rolls)
        recent_window = rolls[-self.rolling_window_size:]
        final_scores = defaultdict(float)

        if len(rolls) >= 2:
            # Markov predictions (40% weight)
            state = tuple(rolls[-2:])
            for num in range(2, 13):
                markov_prob = self.markov_chain.get_probability(state, num)
                if markov_prob > 0:
                    final_scores[num] += markov_prob * 0.4

            # Pattern-based predictions (30% weight)
            for num in range(2, 13):
                confidence = self.calculate_pattern_confidence(rolls, num)
                if confidence > 0:
                    final_scores[num] += confidence * 0.3

            # Frequency-based predictions (30% weight)
            freq_analysis = Counter(recent_window)
            total_recent = len(recent_window)
            for num in range(2, 13):
                actual_freq = freq_analysis.get(num, 0) / total_recent if total_recent > 0 else 0
                expected_freq = self.probability_map[num]
                if actual_freq > 0:
                    final_scores[num] += (actual_freq / expected_freq) * 0.3

        if not final_scores:
            return {'primary': 7, 'confidence': 0, 'alternatives': {}}

        # Get top 3 predictions
        top_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        total_strength = sum(score for _, score in top_predictions)

        return {
            'primary': top_predictions[0][0],
            'confidence': top_predictions[0][1] / total_strength if total_strength > 0 else 0,
            'alternatives': {num: score/total_strength for num, score in top_predictions[1:]}
        }

    def get_optimal_bet_size(self, win_prob: float, odds: float, confidence: float) -> float:
        kelly_fraction = self.calculate_kelly_bet(win_prob, odds)
        confidence_adjusted = kelly_fraction * confidence
        max_bet = self.bankroll * (0.03 if self.strategy_type == BettingStrategy.AGGRESSIVE else 0.02)
        return max(round(min(confidence_adjusted * self.bankroll, max_bet)), self.base_unit)

    def _get_enhanced_place_recommendations(self, rolls, analysis, current_point, prediction):
        recommendations = []

        # Ensure we have enough rolls for meaningful analysis
        if len(rolls) < 2:
            return recommendations

        # Consider valid place bet numbers in order of preference
        place_candidates = []
        for num in [6, 8, 5, 9, 4, 10]:  # Ordered by house edge
            if num != current_point:
                # Calculate base confidence from recent patterns
                pattern_confidence = self.calculate_pattern_confidence(rolls, num)

                # Add frequency-based confidence boost
                recent_rolls = rolls[-min(len(rolls), self.rolling_window_size):]
                freq = Counter(recent_rolls)
                expected_freq = self.probability_map[num]
                actual_freq = freq.get(num, 0) / len(recent_rolls)
                freq_boost = min(actual_freq / expected_freq, 2.0) * 0.2

                # Combined confidence score
                confidence = pattern_confidence + freq_boost

                if confidence >= self.confidence_threshold:
                    # Calculate win probability
                    base_prob = self.probability_map[num]
                    win_prob = base_prob / (base_prob + self.probability_map[7])

                    # Calculate optimal bet size
                    optimal_bet = self.get_optimal_bet_size(
                        win_prob,
                        self.place_bet_odds[num],
                        confidence
                    )

                    if optimal_bet >= self.base_unit:
                        place_candidates.append((num, confidence, optimal_bet))

        # Sort by confidence and bet size
        place_candidates.sort(key=lambda x: (-x[1], -x[2], -1 if x[0] in [6, 8] else 0))

        # Generate recommendations
        for num, confidence, bet_amount in place_candidates[:self.max_place_bets]:
            recommendations.append({
                'bet': f'Place {num}',
                'amount': round(bet_amount),
                'reason': f'Pattern detected (Confidence: {confidence:.1%})',
                'confidence': confidence,
                'rules': [
                    f'Pays {self.place_bet_odds[num]}:1',
                    'Press after two hits',
                    'Remove on seven-out',
                    'Can be turned off on come out rolls',
                    f'Expected value: ${round(bet_amount * self.place_bet_odds[num] * confidence, 2)}'
                ]
            })

        return recommendations

    def recommend_bets(self, rolls, button_placement=None):
        self.button_state.set_button(button_placement)
        self.button_state.set_current_roll(rolls[-1] if rolls else None)

        state = self.button_state.get_state()
        prediction = self.predict_next_roll(rolls)
        analysis = self.analyze_patterns(rolls)

        recommendations = []

        # Pass Line strategy
        if state['on_come_out']:
            # On come out roll
            win_prob = (8/36) / (8/36 + 4/36)  # P(win) / (P(win) + P(lose))
            optimal_bet = max(round(self.base_unit), 1)  # Simplified for come out

            recommendations.append({
                'bet': 'Pass Line',
                'amount': optimal_bet,
                'reason': 'Come out roll - button is OFF',
                'confidence': 1.0,
                'rules': [
                    'Wins on natural (7, 11)',
                    'Loses on craps (2, 3, 12)',
                    f'Take {self.odds_multiplier}x odds when point established',
                    'Wait for button to move OFF before new Pass Line bet'
                ]
            })
        elif state['button'] != 'OFF':
            # Point is established
            point = state['button']
            win_prob = self.probability_map[point] / (self.probability_map[point] + self.probability_map[7])
            optimal_odds = max(round(self.base_unit * self.odds_multiplier), 1)

            recommendations.append({
                'bet': f'Pass Line Odds on {point}',
                'amount': optimal_odds,
                'reason': f'Point is {point} (Button ON)',
                'confidence': 1.0,
                'rules': [
                    f'Pays {self.true_odds[point]}:1',
                    'No house edge on odds',
                    'Remove on seven-out',
                    f'Waiting for {point} or 7'
                ]
            })

        # Place bets when appropriate
        if not state['on_come_out']:
            place_recommendations = self._get_enhanced_place_recommendations(
                rolls, analysis, state['button'], prediction
            )
            recommendations.extend(place_recommendations)

        return {
            'table_state': state,
            'prediction': prediction,
            'recommendations': recommendations[:self.max_place_bets + 1],
            'bankroll_info': {
                'starting': self.bankroll,
                'base_unit': max(round(self.base_unit), 1),
                'strategy': self.strategy_type.value,
                'stop_loss': round(self.bankroll * 0.7)
            },
            'button_info': {
                'position': state['button'],
                'can_place_bets': state['can_place_bet'],
                'on_come_out': state['on_come_out']
            },
            'pattern_analysis': {
                'streaks': analysis['streaks'],
                'next_roll_confidence': prediction['confidence']
            }
        }

    def update_bankroll(self, amount: float):
        """Update bankroll after wins/losses"""
        self.bankroll += amount
        self._init_strategy_parameters()  # Recalibrate based on new bankroll

# Example usage
if __name__ == "__main__":
    # Sample roll sequence
    rolls = [8, 7, 8, 4, 7, 4, 8, 10, 5, 8, 5, 5, 7, 5, 6, 7, 9,
             9, 10, 9, 3, 10, 4, 3, 9, 7, 12,
             5, 7, 11, 10, 10, 8, 9, 8, 6, 7,
             10, 7, 9, 8, 9, 8, 4, 6, 4, 7, 8]

    # Initialize analyzer with aggressive strategy
    analyzer = EnhancedCrapsAnalyzer(bankroll=400, strategy_type=BettingStrategy.AGGRESSIVE)

    # Test with button on 8
    button = 8
    result = analyzer.recommend_bets(rolls, button)

    print(f"\nCraps Analysis - Button on {button}")
    print("=" * 50)

    print("\nTable State:")
    print(f"Button Position: {result['button_info']['position']}")
    print(f"Come Out Roll: {result['button_info']['on_come_out']}")
    print(f"Can Place Bets: {result['button_info']['can_place_bets']}")

    if result['prediction']:
        print(f"\nNext Roll Prediction:")
        print(f"Primary: {result['prediction']['primary']}")
        print(f"Confidence: {result['prediction']['confidence']:.1%}")
        print("Alternatives:", result['prediction']['alternatives'])

    print("\nRecommended Bets:")
    for i, bet in enumerate(result['recommendations'], 1):
        print(f"\n{i}. {bet['bet']}")
        print(f"   Amount: ${bet['amount']}")
        print(f"   Reason: {bet['reason']}")
        print(f"   Confidence: {bet.get('confidence', 1.0):.1%}")
        print("   Rules:")
        for rule in bet['rules']:
            print(f"   - {rule}")

    print(f"\nBankroll Management:")
    print(f"Starting Bankroll: ${result['bankroll_info']['starting']}")
    print(f"Base Unit: ${result['bankroll_info']['base_unit']}")
    print(f"Strategy: {result['bankroll_info']['strategy']}")
    print(f"Stop Loss: ${result['bankroll_info']['stop_loss']}")
