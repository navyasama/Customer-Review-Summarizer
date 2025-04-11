from transformers import pipeline
import re
import logging
from typing import List, Dict, Union, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device="mps",
                truncation=True
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Could not initialize sentiment analyzer")

        # Complete keyword banks with all terms
        self.keyword_banks = {
            'positive': {
                # English
                'excellent': 1.4, 'amazing': 1.4, 'perfect': 1.5, 'awesome': 1.4, 'outstanding': 1.5,
                'fantastic': 1.4, 'superb': 1.5, 'incredible': 1.4, 'phenomenal': 1.6, 'stellar': 1.5,
                'brilliant': 1.5, 'exceptional': 1.5, 'marvelous': 1.4, 'wonderful': 1.4, 'splendid': 1.4,
                'terrific': 1.4, 'fabulous': 1.4, 'remarkable': 1.4, 'stupendous': 1.5, 'first-rate': 1.4,
                'top-notch': 1.5, 'premium': 1.4, 'high-end': 1.3, 'luxury': 1.3, 'flawless': 1.5,
                'impeccable': 1.5, 'pristine': 1.4, 'spotless': 1.3, 'immaculate': 1.4, 'perfectly': 1.4,
                'beautifully': 1.3, 'exquisitely': 1.4, 'superbly': 1.4, 'masterpiece': 1.6, 'gem': 1.4,
                'treasure': 1.4, 'delight': 1.4, 'joy': 1.4, 'pleasure': 1.3, 'satisfaction': 1.3,
                'thrilled': 1.4, 'ecstatic': 1.5, 'overjoyed': 1.5, 'elated': 1.4, 'blissful': 1.4,
                'content': 1.3, 'grateful': 1.3, 'impressed': 1.4, 'astonished': 1.5, 'astounded': 1.5,
                'recommend': 1.4, 'endorse': 1.4, 'praise': 1.4, 'commend': 1.4, 'applaud': 1.4,
                'value': 1.3, 'worth': 1.4, 'bargain': 1.3, 'deal': 1.3, 'steal': 1.4,
                'durable': 1.4, 'long-lasting': 1.4, 'reliable': 1.4, 'sturdy': 1.4, 'well-built': 1.4,
                'high-quality': 1.4, 'premium': 1.4, 'professional': 1.3, 'polished': 1.3, 'refined': 1.3,
                'efficient': 1.4, 'effective': 1.4, 'powerful': 1.4, 'speedy': 1.3, 'quick': 1.3,
                'smooth': 1.4, 'seamless': 1.4, 'intuitive': 1.4, 'user-friendly': 1.4, 'responsive': 1.4,
                'accurate': 1.4, 'precise': 1.4, 'consistent': 1.4, 'dependable': 1.4, 'trustworthy': 1.4,
                
                # Hinglish
                'badhiya': 1.6, 'mast': 1.5, 'jhakaas': 1.7, 'sahi hai': 1.5, 'waah': 1.6,
                'kamaal': 1.6, 'awesome hai': 1.6, 'best hai': 1.6, 'loved it': 1.6, 'superb hai': 1.6,
                'mind blowing': 1.7, 'ekdum sahi': 1.6, 'killer hai': 1.7, 'bohot achha': 1.6, 'sahi product': 1.5,
                'value for money': 1.5, 'paise vasool': 1.6, 'top class': 1.6, 'first class': 1.6, 'number one': 1.6,
                'dhamakedar': 1.7, 'zabardast': 1.7, 'shaandaar': 1.7, 'laajawaab': 1.7, 'behtareen': 1.7,
                'umda': 1.6, 'nayaab': 1.7, 'aag lagadi': 1.7, 'rocking': 1.6, 'hit hai': 1.6,
                'pasand aaya': 1.6, 'dil khush': 1.6, 'jannat': 1.7, 'khushi': 1.6, 'pyaara': 1.5,
                'sundar': 1.5, 'accha laga': 1.6, 'maza aa gaya': 1.7, 'dil se': 1.6, 'dhadkan': 1.6,
                'jaan': 1.7, 'jigar': 1.7, 'dil chahta hai': 1.7, 'jeene ka maza': 1.7, 'jashn': 1.6,
                'khushiyan': 1.6, 'khushnuma': 1.6, 'rangin': 1.6, 'roshan': 1.6, 'chamakdar': 1.6
            },
            'negative': {
                # English
                'terrible': 1.6, 'awful': 1.6, 'horrible': 1.6, 'disgusting': 1.7, 'revolting': 1.7,
                'appalling': 1.7, 'atrocious': 1.7, 'abysmal': 1.7, 'dreadful': 1.6, 'ghastly': 1.7,
                'hideous': 1.6, 'repulsive': 1.7, 'vile': 1.7, 'wretched': 1.6, 'unacceptable': 1.6,
                'deplorable': 1.7, 'disgraceful': 1.7, 'shameful': 1.7, 'contemptible': 1.7, 'despicable': 1.8,
                'lousy': 1.6, 'pathetic': 1.7, 'pitiful': 1.6, 'sorry': 1.6, 'useless': 1.7,
                'worthless': 1.7, 'garbage': 1.8, 'trash': 1.8, 'rubbish': 1.7, 'junk': 1.7,
                'crap': 1.8, 'broken': 1.7, 'defective': 1.7, 'faulty': 1.7, 'malfunctioning': 1.7,
                'nonfunctional': 1.7, 'damaged': 1.7, 'ruined': 1.7, 'destroyed': 1.8, 'unusable': 1.7,
                'unreliable': 1.7, 'inconsistent': 1.6, 'unstable': 1.6, 'crashing': 1.7, 'freezing': 1.7,
                'glitchy': 1.7, 'buggy': 1.7, 'slow': 1.6, 'laggy': 1.7, 'unresponsive': 1.7,
                'disappointing': 1.7, 'unsatisfactory': 1.7, 'underwhelming': 1.7, 'mediocre': 1.6, 'subpar': 1.7,
                'inferior': 1.7, 'poor': 1.7, 'cheap': 1.7, 'flimsy': 1.7, 'shoddy': 1.7,
                'tacky': 1.6, 'low-quality': 1.7, 'unprofessional': 1.7, 'amateurish': 1.7, 'sloppy': 1.7,
                'careless': 1.7, 'negligent': 1.7, 'incompetent': 1.8, 'irresponsible': 1.8, 'rude': 1.7,
                'impolite': 1.6, 'disrespectful': 1.7, 'insulting': 1.8, 'offensive': 1.8, 'unpleasant': 1.6,
                'nasty': 1.7, 'hostile': 1.7, 'aggressive': 1.7, 'hateful': 1.8, 'toxic': 1.8,
                'harmful': 1.8, 'dangerous': 1.9, 'hazardous': 1.9, 'unsafe': 1.9, 'scam': 1.9,
                'fraud': 1.9, 'fake': 1.9, 'counterfeit': 1.9, 'misleading': 1.8, 'deceptive': 1.8,
                'dishonest': 1.9, 'unethical': 1.9, 'immoral': 1.9, 'corrupt': 1.9, 'waste': 1.7,
                
                # Hinglish
                'bekaar': 1.8, 'ghatiya': 1.9, 'nikamma': 1.8, 'bakwas': 1.8, 'bura': 1.7,
                'kharab': 1.8, 'kachra': 1.9, 'latka hua': 1.7, 'barbaad': 1.9, 'dhoka': 1.9,
                'fraud hai': 1.9, 'scam hai': 2.0, 'cheat kiya': 2.0, 'loot liya': 2.0, 'paise barbaad': 2.0,
                'time waste': 1.8, 'dimag kharab': 1.9, 'paagal bana diya': 2.0, 'bewakoof banaaya': 2.0, 'dikkat hai': 1.8,
                'problem hai': 1.8, 'tension hai': 1.8, 'pareshan': 1.8, 'niraash': 1.8, 'udaas': 1.7,
                'dil tuta': 1.9, 'ro diya': 1.9, 'dard hua': 1.8, 'takleef hui': 1.8, 'gussa aaya': 1.8,
                'naraz': 1.8, 'khafa': 1.8, 'naaraaz': 1.8, 'bezaar': 1.8, 'tang': 1.7,
                'pareshan': 1.8, 'thaka': 1.7, 'dukhi': 1.8, 'afsoos': 1.8, 'pachtawa': 1.8,
                'laanat': 2.0, 'shraap': 2.0, 'abshar': 2.0, 'phat gayi': 1.9, 'fat gaya': 1.9,
                'toot gaya': 1.9, 'gir gaya': 1.8, 'khatam': 1.9, 'khatam ho gaya': 2.0, 'bura hai': 1.8
            },
            'neutral': {
                # English
                'okay': 0.8, 'decent': 0.8, 'average': 0.7, 'mediocre': 0.7, 'moderate': 0.7,
                'ordinary': 0.7, 'common': 0.7, 'typical': 0.7, 'standard': 0.7, 'regular': 0.7,
                'usual': 0.7, 'normal': 0.7, 'fair': 0.8, 'reasonable': 0.8, 'acceptable': 0.8,
                'adequate': 0.8, 'sufficient': 0.7, 'passable': 0.8, 'tolerable': 0.8, 'bearable': 0.7,
                'so-so': 0.7, 'meh': 0.6, 'middling': 0.7, 'indifferent': 0.6, 'unremarkable': 0.7,
                'forgettable': 0.6, 'undistinguished': 0.6, 'run-of-the-mill': 0.7, 'garden-variety': 0.7, 'vanilla': 0.7,
                'basic': 0.7, 'simple': 0.7, 'plain': 0.7, 'unexceptional': 0.7, 'uninspired': 0.6,
                'unimpressive': 0.6, 'lackluster': 0.6, 'bland': 0.6, 'dull': 0.6, 'tedious': 0.6,
                'monotonous': 0.6, 'repetitive': 0.6, 'predictable': 0.6, 'conventional': 0.7, 'traditional': 0.7,
                'generic': 0.7, 'commonplace': 0.7, 'everyday': 0.7, 'workable': 0.8, 'functional': 0.8,
                'practical': 0.8, 'utilitarian': 0.7, 'serviceable': 0.8, 'usable': 0.8, 'adequate': 0.8,
                'satisfactory': 0.8, 'sufficient': 0.8, 'competent': 0.8, 'decent': 0.8, 'respectable': 0.8,
                'presentable': 0.8, 'tolerable': 0.8, 'passable': 0.8, 'uninspiring': 0.6, 'unexciting': 0.6,
                'unmemorable': 0.6, 'forgettable': 0.6, 'undistinguished': 0.6, 'unexceptional': 0.6, 'unremarkable': 0.6,
                'middle-of-the-road': 0.7, 'neither here nor there': 0.6, 'take it or leave it': 0.6, 'nothing special': 0.6, 
                'nothing to write home about': 0.6, 'could be better': 0.7, 'could be worse': 0.7, 'room for improvement': 0.7,
                'has potential': 0.7, 'work in progress': 0.7, 'getting there': 0.7, 'on its way': 0.7, 'almost there': 0.7,
                'not bad': 0.8, 'not great': 0.7, 'not terrible': 0.8, 'not horrible': 0.8, 'not awful': 0.8,
                'not the worst': 0.8, 'not the best': 0.7, 'just okay': 0.7, 'passing grade': 0.7, 'barely acceptable': 0.6,
                
                # Hinglish
                'theek thaak': 0.7, 'chalta hai': 0.6, 'okayish': 0.7, 'thik hai': 0.7, 'normal hai': 0.7,
                'aam': 0.7, 'sadharan': 0.7, 'maamuli': 0.6, 'khaas nahi': 0.6, 'kuch khaas nahi': 0.6,
                'aacha bhi nahi bura bhi nahi': 0.5, 'na ghar ka na ghaat ka': 0.5, 'na idhar ka na udhar ka': 0.5, 'beech ka': 0.6, 'madhyam': 0.7,
                'satisfactory hai': 0.8, 'expectation se kam': 0.6, 'zyaada kuch nahi': 0.6, 'bas chalne layak': 0.6, 'improvement chahiye': 0.7,
                'thoda aur aacha ho sakta tha': 0.7, 'nahi bura': 0.7, 'timepass': 0.6, 'khaali paisa kharcha': 0.6, 'paise ka sahi': 0.7,
                'sasta': 0.7, 'majboori': 0.6, 'kaam chalao': 0.6, 'jaisa taisa': 0.6, 'jugaad': 0.7,
                'adjust': 0.7, 'compromise': 0.6, 'manage': 0.7, 'chalau': 0.6, 'aadat daal lo': 0.6,
                'aadat pad gayi': 0.6, 'aadat ho gaya': 0.6, 'aadat se majboor': 0.6, 'majboori ka naam': 0.6, 'kismat': 0.6,
                'naseeb': 0.6, 'bhagwan bharose': 0.5, 'jo hota hai': 0.5, 'jo hoga dekha jayega': 0.5, 'chalta hai yaar': 0.6,
                'koi farak nahi': 0.6, 'farak nahi padta': 0.6, 'kya fayda': 0.5, 'kya kar sakte hai': 0.5, 'chhod do': 0.5
            }
        }

        # Complete slang mappings
        self.slang_mappings = {
            'positive': {
                'badiya': 'badhiya', 
                'mazaa': 'mast', 
                'jhakkas': 'jhakaas',
                'vah': 'waah',
                'accha': 'sahi hai',
                'gr8': 'great',
                'luv': 'love',
                'awsm': 'awesome',
                'lit': 'awesome',
                'fab': 'fabulous'
            },
            'negative': {
                'bakwas': 'bekaar',
                'nikamma': 'kharab',
                'fraud': 'scam hai',
                'crap': 'garbage',
                'sucks': 'terrible',
                'bs': 'bakwas',
                'wtf': 'terrible',
                'uff': 'frustrating',
                'ugh': 'disgusting'
            },
            'neutral': {
                'thik thak': 'theek thaak',
                'ok ok': 'okayish',
                'mid': 'average',
                'meh': 'chalta hai',
                'avg': 'average',
                'mehh': 'mediocre',
                'whatever': 'indifferent'
            }
        }

        self._compile_patterns()

    def _compile_patterns(self):
        """Safe pattern compilation with validation"""
        expanded_keywords = defaultdict(dict)
        
        for sentiment, keywords in self.keyword_banks.items():
            expanded_keywords[sentiment].update(keywords)
            
            if sentiment in self.slang_mappings:
                for slang, canonical in self.slang_mappings[sentiment].items():
                    if canonical in keywords:
                        expanded_keywords[sentiment][slang] = keywords[canonical]
                    else:
                        logger.warning(f"Slang mapping skipped - canonical term '{canonical}' not found for '{slang}'")

        self.patterns = {
            sentiment: re.compile(
                r'\b(' + '|'.join(map(re.escape, keywords.keys())) + r')\b',
                re.IGNORECASE
            )
            for sentiment, keywords in expanded_keywords.items()
        }

    def _apply_keyword_weights(self, text: str, base_score: float, label: str) -> Tuple[float, str]:
        """Enhanced weighting with Hinglish priority"""
        text_lower = text.lower()
        hinglish_boost = 1.0
        new_label = label
        
        # Priority handling for strong Hinglish positives
        for word in ['badhiya', 'mast', 'jhakaas', 'paise vasool', 'ekdum sahi']:
            if word in text_lower:
                if new_label != 'POSITIVE':
                    new_label = 'POSITIVE'
                    base_score = max(0.8, base_score)
                hinglish_boost *= 1.6
                break
        
        # Apply standard keyword weights
        weighted_score = base_score * hinglish_boost
        for sentiment, pattern in self.patterns.items():
            for match in pattern.findall(text_lower):
                canonical = self.slang_mappings.get(sentiment, {}).get(match.lower(), match.lower())
                weight = self.keyword_banks[sentiment].get(canonical, 1.0)
                weighted_score *= weight if sentiment == new_label else 1/weight
        
        return (min(1.0, max(0.1, weighted_score)), new_label)

    def _contains_negation(self, text: str) -> bool:
        """Check for negations that flip sentiment"""
        negations = {'not', 'no', 'nahi', 'never', 'nope', 'na', 'without', 'nahi hai', 'nahi tha'}
        return any(re.search(rf'\b{neg}\b', text.lower()) for neg in negations)

    def analyze_sentiments(self, reviews: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Final sentiment analysis with all keywords"""
        results = []
        
        for review in [r for r in reviews if isinstance(r, str) and r.strip()]:
            try:
                # Base prediction
                prediction = self.model(review[:512], truncation=True)[0]
                label = prediction['label']
                score = prediction['score']
                
                # Apply keyword adjustments
                adjusted_score, updated_label = self._apply_keyword_weights(review, score, label)
                label = updated_label
                
                # Handle negations
                if self._contains_negation(review):
                    adjusted_score = 1 - adjusted_score
                    label = 'NEGATIVE' if label == 'POSITIVE' else 'POSITIVE'
                
                # Final confidence adjustments
                if label == 'POSITIVE' and any(phrase in review.lower() for phrase in ['badhiya', 'mast', 'jhakaas']):
                    adjusted_score = max(adjusted_score, 0.9)
                if label == 'NEUTRAL' and any(phrase in review.lower() for phrase in ['chalta hai', 'timepass']):
                    adjusted_score = min(adjusted_score, 0.6)
                
                results.append({
                    'review': review,
                    'sentiment': label,
                    'confidence': round(adjusted_score, 4),
                    'keywords': list(set(
                        self.slang_mappings.get('positive', {}).get(match.lower(), match.lower())
                        for sentiment in self.patterns.values() 
                        for match in sentiment.findall(review.lower())
                    ))
                })
                
            except Exception as e:
                logger.warning(f"Skipping review: {str(e)}")
        
        return results

# Initialize analyzer
try:
    analyzer = SentimentAnalyzer()
    analyze_sentiments = analyzer.analyze_sentiments
except Exception as e:
    logger.error(f"Analyzer initialization failed: {str(e)}")
    raise