from flask import Flask, request, jsonify, render_template
from summarizer import analyze_sentiments  # Only import the function we need
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    """Render the main interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint for sentiment analysis"""
    try:
        # Input validation
        reviews_raw = request.form.get('reviews', '').strip()
        if not reviews_raw:
            return jsonify({'error': 'No reviews provided'}), 400

        reviews = [r.strip() for r in reviews_raw.split('\n') if r.strip()]
        if not reviews:
            return jsonify({'error': 'No valid reviews found'}), 400

        # Process reviews
        logger.info(f"Analyzing {len(reviews)} reviews")
        sentiment_results = analyze_sentiments(reviews)

        # Prepare response
        pos_count = sum(1 for r in sentiment_results if r['sentiment'] == 'POSITIVE')
        neg_count = len(sentiment_results) - pos_count

        return jsonify({
            'sentiments': sentiment_results,
            'stats': {
                'total': len(sentiment_results),
                'positive': pos_count,
                'negative': neg_count
            }
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=5001,
        debug=True,
        threaded=True
    )

