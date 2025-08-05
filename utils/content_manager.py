import random
from typing import Dict, List, Tuple

class ContentManager:
    def __init__(self):
        self.difficulty_levels = {
            'beginner': {
                'cefr_range': ['A1', 'A2'],
                'sentence_length': (10, 20),
                'complexity': 'simple'
            },
            'intermediate': {
                'cefr_range': ['B1', 'B2'],
                'sentence_length': (15, 35),
                'complexity': 'moderate'
            },
            'advanced': {
                'cefr_range': ['C1', 'C2'],
                'sentence_length': (25, 50),
                'complexity': 'complex'
            }
        }
        
        self.topics = {
            'business': {
                'sentences': [
                    "The quarterly revenue report indicates a significant increase in market share across all regions.",
                    "Effective communication skills are essential for successful team collaboration and project management.",
                    "Strategic planning requires careful analysis of market trends and competitive positioning.",
                    "Customer satisfaction metrics demonstrate the importance of quality service delivery.",
                    "Innovation in product development drives sustainable competitive advantage in the marketplace."
                ]
            },
            'academic': {
                'sentences': [
                    "The research methodology employed quantitative and qualitative approaches to ensure comprehensive data collection.",
                    "Statistical analysis revealed significant correlations between variables in the experimental group.",
                    "Theoretical frameworks provide essential foundations for understanding complex phenomena.",
                    "Peer-reviewed literature supports the hypothesis regarding environmental impact assessment.",
                    "Methodological rigor ensures the validity and reliability of research findings."
                ]
            },
            'casual': {
                'sentences': [
                    "I really enjoyed the movie we watched last night, especially the unexpected plot twist.",
                    "The weather has been absolutely beautiful this week, perfect for outdoor activities.",
                    "I'm thinking about trying that new restaurant downtown that everyone's been talking about.",
                    "The concert was amazing, the band played all my favorite songs from their latest album.",
                    "I can't believe how quickly this year has gone by, it feels like summer just started."
                ]
            },
            'technology': {
                'sentences': [
                    "Artificial intelligence algorithms are revolutionizing data processing and decision-making systems.",
                    "Blockchain technology provides secure and transparent transaction verification mechanisms.",
                    "Machine learning models require extensive training data to achieve optimal performance accuracy.",
                    "Cloud computing infrastructure enables scalable and flexible resource allocation.",
                    "Cybersecurity protocols are essential for protecting sensitive digital information."
                ]
            },
            'environment': {
                'sentences': [
                    "Climate change mitigation requires coordinated global efforts to reduce carbon emissions.",
                    "Renewable energy sources provide sustainable alternatives to fossil fuel consumption.",
                    "Biodiversity conservation efforts protect essential ecosystem services and species survival.",
                    "Environmental sustainability practices promote responsible resource management.",
                    "Green technology innovations drive economic growth while protecting natural resources."
                ]
            }
        }
        
        self.interactive_exercises = {
            'tongue_twisters': [
                "Peter Piper picked a peck of pickled peppers.",
                "She sells seashells by the seashore.",
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
                "The quick brown fox jumps over the lazy dog.",
                "Unique New York, New York's unique."
                "Fred fed Ted bread and Ted fed Fred bread.",
                "Six slippery snails slid slowly seaward.",
                "Brisk brave brigadiers brandish broad bright blades.",
                "Truly rural, truly rural, truly rural.",
                "A proper copper coffee pot.",
                "I saw Susie sitting in a shoeshine shop.",
                "Black bug’s blood, black bug’s blood, black bug’s blood.",
                "Thin sticks, thick bricks, thick sticks, thin bricks.",
                "Can you can a can as a canner can can a can?",
                "Lesser leather never weathered wetter weather better."
            ],
            'minimal_pairs': [
                ("ship", "sheep"),
                ("bit", "beat"),
                ("cot", "caught"),
                ("pull", "pool"),
                ("full", "fool")
            ],
            'stress_patterns': [
                "REcord (noun) vs reCORD (verb)",
                "PREsent (noun) vs preSENT (verb)",
                "CONtract (noun) vs conTRACT (verb)",
                "PROject (noun) vs proJECT (verb)",
                "IMport (noun) vs imPORT (verb)"
            ]
        }
    
    def get_sentence_by_difficulty(self, difficulty: str = 'intermediate') -> str:
        """Get a sentence based on difficulty level"""
        if difficulty not in self.difficulty_levels:
            difficulty = 'intermediate'
        
        # Get sentences from all topics for the difficulty level
        all_sentences = []
        for topic, content in self.topics.items():
            all_sentences.extend(content['sentences'])
        
        # Filter by difficulty (simplified approach)
        if difficulty == 'beginner':
            # Use shorter, simpler sentences
            filtered_sentences = [s for s in all_sentences if len(s.split()) <= 15]
        elif difficulty == 'advanced':
            # Use longer, more complex sentences
            filtered_sentences = [s for s in all_sentences if len(s.split()) >= 25]
        else:
            # Intermediate - use all sentences
            filtered_sentences = all_sentences
        
        return random.choice(filtered_sentences) if filtered_sentences else random.choice(all_sentences)
    
    def get_sentence_by_topic(self, topic: str = 'business') -> str:
        """Get a sentence from a specific topic"""
        if topic not in self.topics:
            topic = 'business'
        
        return random.choice(self.topics[topic]['sentences'])
    
    def get_interactive_exercise(self, exercise_type: str = 'tongue_twisters') -> Dict:
        """Get an interactive exercise"""
        if exercise_type not in self.interactive_exercises:
            exercise_type = 'tongue_twisters'
        
        if exercise_type == 'minimal_pairs':
            pair = random.choice(self.interactive_exercises[exercise_type])
            return {
                'type': 'minimal_pairs',
                'content': pair,
                'instruction': f"Practice the difference between '{pair[0]}' and '{pair[1]}'"
            }
        elif exercise_type == 'stress_patterns':
            pattern = random.choice(self.interactive_exercises[exercise_type])
            return {
                'type': 'stress_patterns',
                'content': pattern,
                'instruction': "Practice the stress pattern difference"
            }
        else:
            content = random.choice(self.interactive_exercises[exercise_type])
            return {
                'type': 'tongue_twisters',
                'content': content,
                'instruction': "Practice this tongue twister slowly, then increase speed"
            }
    
    def get_all_topics(self) -> List[str]:
        """Get list of all available topics"""
        return list(self.topics.keys())
    
    def get_all_difficulties(self) -> List[str]:
        """Get list of all available difficulty levels"""
        return list(self.difficulty_levels.keys())
    
    def get_all_exercise_types(self) -> List[str]:
        """Get list of all available exercise types"""
        return list(self.interactive_exercises.keys()) 