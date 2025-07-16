"""
Interactive tool for reviewing and correcting financial news sentiment labels
with progress tracking and automatic resume functionality
"""

import pandas as pd
import os
import sys
import argparse
import re
import json
from datetime import datetime

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style, Back
    has_colorama = True
    init()
except ImportError:
    has_colorama = False

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored_text(text, sentiment=None):
    """Print text with color based sentiment"""
    if not has_colorama:
        print(text)
        return
        
    if sentiment == "positive":
        print(Fore.GREEN + text + Style.RESET_ALL)
    elif sentiment == "negative":
        print(Fore.RED + text + Style.RESET_ALL)
    elif sentiment == "neutral":
        print(Fore.BLUE + text + Style.RESET_ALL)
    else:
        print(text)

def get_progress_file_path(input_file):
    """Get the path for the progress file"""
    base_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(base_dir, f".{base_name}_labeling_progress.json")

def save_progress(input_file, last_index, total_reviewed, total_changed, user_login):
    """Save the current progress to a file"""
    progress_file = get_progress_file_path(input_file)
    progress_data = {
        'user': user_login,
        'last_index': last_index,
        'total_reviewed': total_reviewed,
        'total_changed': total_changed,
        'timestamp': datetime.now().isoformat(),
        'file_path': input_file
    }
    
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def load_progress(input_file, user_login):
    """Load previous progress from file"""
    progress_file = get_progress_file_path(input_file)
    
    if not os.path.exists(progress_file):
        return None
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        # Check if the progress belongs to the current user
        if progress_data.get('user') != user_login:
            print(f"Found progress for different user: {progress_data.get('user')}")
            return None
            
        return progress_data
    except Exception as e:
        print(f"Warning: Could not load progress: {e}")
        return None

def highlight_financial_context(text):
    """Highlight financial contexts to aid sentiment judgment"""
    if not has_colorama:
        return text
    
    # Financial contexts that change meaning
    contexts = [
        # INFLATION CONTEXTS
        # Inflation decreasing = POSITIVE
        (r'(inflation.{0,30}(ralentit|baisse|diminue|recul))', Fore.GREEN),
        (r'((baisse|diminution|recul|ralentit).{0,30}inflation)', Fore.GREEN),
        (r'(hausse des prix.{0,30}ralentit)', Fore.GREEN),
        
        # Inflation increasing = NEGATIVE
        (r'(inflation.{0,30}(hausse|augmente|accélère))', Fore.RED),
        (r'((hausse|augmentation|accélère).{0,30}inflation)', Fore.RED),
        (r'(inflation.{0,30}record)', Fore.RED),
        
        # BUSINESS CONTEXTS
        # Negative business language
        (r'(baisse.{0,30}performances)', Fore.RED),
        (r'(contexte.{0,10}(tendu|difficile))', Fore.RED),
        (r'(marché.{0,10}saturé)', Fore.RED),
        (r'(dynamique.{0,10}essoufflée)', Fore.RED),
        (r'(difficultés.{0,20}secteur)', Fore.RED),
        (r'(peinent.{0,30}objectifs)', Fore.RED),
        (r'(fragilise|fragile|fragilité)', Fore.RED),
        
        # PRICES & INFLATION (price up = negative, price down = positive)
        (r'((hausse|augmentation).{0,20}(prix|tarif|coût))', Fore.RED),
        (r'((prix|tarifs).{0,20}(hausse|augmentent|augmentation))', Fore.RED),
        (r'((baisse|diminution|recul|repli).{0,20}(prix|tarif|coût))', Fore.GREEN),
        (r'((prix|tarifs).{0,20}(baisse|diminuent|recul))', Fore.GREEN),
        
        # MARKET (market up = positive, market down = negative)
        (r'((masi|madex|marché).{0,30}(baisse|recul|chute|repli))', Fore.RED),
        (r'(séance[s]? négative[s]?)', Fore.RED),
        (r'(clôture en baisse)', Fore.RED),
        (r'((masi|madex|marché).{0,30}(hausse|progression|augmentation))', Fore.GREEN),
        (r'(séance[s]? positive[s]?)', Fore.GREEN),
        (r'(clôture en hausse)', Fore.GREEN),
        
        # COMPANY PERFORMANCE
        (r'((baisse|diminution|recul|chute).{0,20}(bénéfice|résultat|profit|revenu|chiffre d\'affaires))', Fore.RED),
        (r'((perte|déficit).{0,40}(entreprise|société|groupe))', Fore.RED),
        (r'((hausse|augmentation|progression|croissance).{0,20}(bénéfice|résultat|profit|revenu|chiffre d\'affaires))', Fore.GREEN),
        (r'((bénéfice|profit).{0,40}(entreprise|société|groupe))', Fore.GREEN),
    ]
    
    # Highlight each context
    for pattern, color in contexts:
        text = re.sub(f'({pattern})', f'{color}\\1{Style.RESET_ALL}', text, flags=re.IGNORECASE)
    
    # Highlight percentages
    text = re.sub(r'(\+\d+[,.]\d+\s*%)', f'{Fore.GREEN}\\1{Style.RESET_ALL}', text)
    text = re.sub(r'(-\d+[,.]\d+\s*%)', f'{Fore.RED}\\1{Style.RESET_ALL}', text)
    
    return text

def find_suspicious_articles(df, sentiment_column='sentiment'):
    """Find articles with potentially incorrect sentiment labels"""
    suspicious_indices = []
    
    # Special context-based checks
    for i, row in df.iterrows():
        # Skip rows without sentiment
        if pd.isna(row[sentiment_column]):
            continue
            
        text = str(row['text']).lower() if pd.notna(row['text']) else ""
        title = str(row['title']).lower() if 'title' in df.columns and pd.notna(row['title']) else ""
        sentiment = row[sentiment_column]
        
        full_text = f"{title}. {text}"
        
        # Check for INFLATION context
        inflation_down_patterns = [
            r'inflation.{0,20}(baisse|diminue|recul|ralentit)',
            r'(baisse|diminution|recul|ralentit).{0,20}inflation',
            r'hausse des prix ralentit'
        ]
        
        inflation_up_patterns = [
            r'inflation.{0,20}(hausse|augmente)',
            r'(hausse|augmentation).{0,20}inflation',
            r'hausse des prix'
        ]
        
        # Check for contradictions in inflation context
        for pattern in inflation_down_patterns:
            if re.search(pattern, full_text) and sentiment != 'positive':
                suspicious_indices.append(i)
                break
                
        for pattern in inflation_up_patterns:
            if re.search(pattern, full_text) and sentiment != 'negative':
                suspicious_indices.append(i)
                break
        
        # Check for negative business language
        negative_business_patterns = [
            r'baisse.{0,30}performances',
            r'contexte.{0,10}(tendu|difficile)',
            r'marché.{0,10}saturé',
            r'dynamique.{0,10}essoufflée',
            r'difficultés.{0,20}secteur',
            r'peinent.{0,30}objectifs',
            r'fragilise|fragile|fragilité'
        ]
        
        for pattern in negative_business_patterns:
            if re.search(pattern, full_text) and sentiment != 'negative':
                suspicious_indices.append(i)
                break
        
        # Check for MARKET context
        market_down_patterns = [
            r'(masi|madex|marché).{0,30}(baisse|recul|chute|repli)',
            r'séance négative',
            r'clôture en baisse'
        ]
        
        market_up_patterns = [
            r'(masi|madex|marché).{0,30}(hausse|progression|augmentation)',
            r'séance positive',
            r'clôture en hausse'
        ]
        
        # Check for contradictions in market context
        for pattern in market_down_patterns:
            if re.search(pattern, full_text) and sentiment != 'negative':
                suspicious_indices.append(i)
                break
                
        for pattern in market_up_patterns:
            if re.search(pattern, full_text) and sentiment != 'positive':
                suspicious_indices.append(i)
                break
        
        # Check for negative percentages
        if re.search(r'-\d+[,.]\d+\s*%', full_text) and sentiment != 'negative' and not re.search(r'inflation', full_text):
            suspicious_indices.append(i)
            
        # Check for positive percentages
        if re.search(r'\+\d+[,.]\d+\s*%', full_text) and sentiment != 'positive' and not re.search(r'inflation', full_text):
            suspicious_indices.append(i)
    
    return suspicious_indices

def review_articles(input_file, output_file=None, start_index=None, max_articles=None, 
                    display_processed=False, prioritize_suspicious=True, user_login="khalidAgnaber"):
    """
    Interactive tool to review and correct sentiment labels with progress tracking
    
    Args:
        input_file: Path to the CSV file with articles
        output_file: Path to save corrected articles
        start_index: Index to start reviewing from (None to use saved progress)
        max_articles: Maximum number of articles to review
        display_processed: Whether to show processed text instead of original
        prioritize_suspicious: Whether to review suspicious articles first
        user_login: User login for progress tracking
    """
    # Set default output file if not provided
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_corrected.csv"
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} articles from {input_file}")
    
    # Check if we have sentiment and confidence scores
    has_score = 'sentiment_score' in df.columns
    
    # Load previous progress if no start index specified
    previous_progress = None
    if start_index is None:
        previous_progress = load_progress(input_file, user_login)
        if previous_progress:
            start_index = previous_progress['last_index'] + 1  # Start from next article
            print(f"Resuming from previous session:")
            print(f"  Last reviewed: article {previous_progress['last_index']}")
            print(f"  Total reviewed: {previous_progress['total_reviewed']}")
            print(f"  Total changed: {previous_progress['total_changed']}")
            print(f"  Last session: {previous_progress['timestamp']}")
        else:
            start_index = 0
    
    # Find suspicious articles
    suspicious_indices = []
    if prioritize_suspicious and 'sentiment' in df.columns:
        suspicious_indices = find_suspicious_articles(df)
        print(f"Found {len(suspicious_indices)} articles with potentially incorrect sentiment labels")
        
        # Display some examples of suspicious articles
        if suspicious_indices:
            print("\nExamples of suspicious articles:")
            for i in suspicious_indices[:3]:
                title = df.loc[i, 'title'] if 'title' in df.columns and pd.notna(df.loc[i, 'title']) else ""
                sentiment = df.loc[i, 'sentiment']
                print(f"  - {title} (labeled as {sentiment})")
    
    # Start from the specified index
    start_idx = max(start_index, 0)
    if start_idx > 0:
        print(f"Starting from article {start_idx}")
    
    # Determine text column to display
    text_col = 'processed_text' if display_processed and 'processed_text' in df.columns else 'text'
    
    # Create review order - unlabeled first, then suspicious, then the rest
    if prioritize_suspicious:
        review_order = []
        
        # First add unlabeled articles at or after start_idx
        unlabeled_indices = df[df['sentiment'].isnull()].index.tolist()
        for idx in unlabeled_indices:
            if idx >= start_idx:
                review_order.append(idx)
        
        # Then add suspicious articles at or after start_idx
        for idx in suspicious_indices:
            if idx >= start_idx and idx not in review_order:
                review_order.append(idx)
        
        # Then add other articles at or after start_idx
        for idx in range(start_idx, len(df)):
            if idx not in review_order:
                review_order.append(idx)
    else:
        # Simple sequential order
        review_order = list(range(start_idx, len(df)))
    
    # Apply max_articles limit if specified
    if max_articles:
        review_order = review_order[:max_articles]
        print(f"Will review up to {len(review_order)} articles")
    else:
        print(f"Will review {len(review_order)} articles")
    
    # Show current sentiment distribution
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        print("Current sentiment distribution:")
        for label, count in sentiment_counts.items():
            if pd.isna(label):
                print(f"  Not labeled: {count}")
            else:
                print(f"  {label}: {count}")
    
    input("Press Enter to begin reviewing...")
    
    # Initialize counters
    count = 0
    changed = previous_progress['total_changed'] if previous_progress else 0
    total_reviewed = previous_progress['total_reviewed'] if previous_progress else 0
    
    # Review articles
    try:
        for i in review_order:
            clear_screen()
            suspicious = i in suspicious_indices
            unlabeled = pd.isna(df.loc[i, 'sentiment']) if 'sentiment' in df.columns else True
            
            # Indicate status of this article
            status_indicators = []
            if unlabeled:
                status_indicators.append(f"{Fore.CYAN}UNLABELED{Style.RESET_ALL}")
            if suspicious:
                status_indicators.append(f"{Fore.YELLOW}SUSPICIOUS{Style.RESET_ALL}")
            
            status_str = " ".join(status_indicators) if status_indicators else ""
            print(f"{status_str} Article {count+1}/{len(review_order)} (Index: {i})")
            print("=" * 80)
            
            # Display title if available
            if 'title' in df.columns and pd.notna(df.loc[i, 'title']):
                title = df.loc[i, 'title']
                if has_colorama:
                    title = highlight_financial_context(title)
                print(f"Title: {title}")
                print("-" * 80)
            
            # Display text with highlighted financial context
            text = str(df.loc[i, text_col])
            if has_colorama:
                # Highlight financial context to aid sentiment judgment
                text = highlight_financial_context(text)
            
            # Display text (truncated if very long)
            if len(text) > 1000:
                print(f"Text: {text[:1000]}... [truncated, {len(text)} chars total]")
            else:
                print(f"Text: {text}")
            
            print("-" * 80)
            
            # Display current sentiment if available
            if 'sentiment' in df.columns and pd.notna(df.loc[i, 'sentiment']):
                current_sentiment = df.loc[i, 'sentiment']
                if has_score and pd.notna(df.loc[i, 'sentiment_score']):
                    score = df.loc[i, 'sentiment_score']
                    print(f"Current sentiment: {current_sentiment.upper()} (confidence: {score:.2f})")
                else:
                    print(f"Current sentiment: {current_sentiment.upper()}")
            
            print("=" * 80)
            print("FINANCIAL CONTEXT GUIDE:")
            print(f"{Fore.GREEN}Green{Style.RESET_ALL} text indicates positive financial impact")
            print(f"{Fore.RED}Red{Style.RESET_ALL} text indicates negative financial impact")
            print(f"Remember: Inflation/price decreases are POSITIVE, market decreases are NEGATIVE")
            print("=" * 80)
            
            if 'sentiment' in df.columns and pd.notna(df.loc[i, 'sentiment']):
                print("Keep or change sentiment? (k)eep, (p)ositive, (n)egative, (u)tral, (s)kip, (q)uit")
            else:
                print("Select sentiment: (p)ositive, (n)egative, (u)tral, (s)kip, (q)uit")
            
            # Get user input
            choice = input("> ").strip().lower()
            
            if choice == 'q':
                print("Quitting. Progress saved.")
                break
            elif choice == 's':
                print("Skipping this article.")
            elif choice in ['k', 'keep'] and 'sentiment' in df.columns and pd.notna(df.loc[i, 'sentiment']):
                current_sentiment = df.loc[i, 'sentiment']
                print_colored_text(f"Keeping as {current_sentiment.upper()}", current_sentiment)
            elif choice in ['p', 'positive']:
                if 'sentiment' not in df.columns or df.loc[i, 'sentiment'] != 'positive':
                    changed += 1
                df.loc[i, 'sentiment'] = 'positive'
                print_colored_text("Set to POSITIVE", "positive")
            elif choice in ['n', 'negative']:
                if 'sentiment' not in df.columns or df.loc[i, 'sentiment'] != 'negative':
                    changed += 1
                df.loc[i, 'sentiment'] = 'negative'
                print_colored_text("Set to NEGATIVE", "negative")
            elif choice in ['u', 'neutral']:
                if 'sentiment' not in df.columns or df.loc[i, 'sentiment'] != 'neutral':
                    changed += 1
                df.loc[i, 'sentiment'] = 'neutral'
                print_colored_text("Set to NEUTRAL", "neutral")
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Save after each change
            df.to_csv(output_file, index=False)
            if choice not in ['s', 'skip']:
                print(f"Progress saved to {output_file}")
            
            # Update counters
            count += 1
            total_reviewed += 1
            
            # Save progress after every article
            save_progress(input_file, i, total_reviewed, changed, user_login)
    
    except KeyboardInterrupt:
        print("\nReview interrupted. Progress saved.")
    
    # Final stats
    print(f"\nReview complete. Reviewed {count} articles in this session.")
    print(f"Total articles reviewed: {total_reviewed}")
    print(f"Total changes made: {changed}")
    
    # Show updated distribution
    if 'sentiment' in df.columns:
        new_sentiment_counts = df['sentiment'].value_counts().to_dict()
        print("Updated sentiment distribution:")
        for label, count in new_sentiment_counts.items():
            if pd.isna(label):
                print(f"  Not labeled: {count}")
            else:
                print(f"  {label}: {count}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive tool for reviewing sentiment labels")
    parser.add_argument("input_file", help="Path to the CSV file with labeled articles")
    parser.add_argument("-o", "--output_file", help="Path to save corrected articles")
    parser.add_argument("-s", "--start", type=int, help="Index to start reviewing from (overrides saved progress)")
    parser.add_argument("-m", "--max", type=int, help="Maximum number of articles to review")
    parser.add_argument("-p", "--processed", action="store_true", 
                        help="Display processed text instead of original")
    parser.add_argument("--no-suspicious", action="store_true",
                        help="Don't prioritize suspicious articles")
    parser.add_argument("-u", "--user", default="khalidAgnaber",
                        help="User login for progress tracking")
    
    args = parser.parse_args()
    review_articles(
        args.input_file, 
        args.output_file, 
        args.start, 
        args.max, 
        args.processed,
        not args.no_suspicious,
        args.user
    )