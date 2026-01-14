# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "plotly",
#     "kaleido",
#     "scikit-learn",
#     "wordcloud",
# ]
# ///

import json
import glob
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "/media/chirag/Datasets/arxiv-data-exploration/openAlex"
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expanded Stop Words List (Custom + NLTK-ish)
STOP_WORDS = set([
    "the", "and", "of", "to", "in", "a", "for", "with", "on", "is", "by", "an", "as", 
    "this", "that", "are", "from", "be", "at", "or", "it", "we", "which", "can",
    "using", "based", "propose", "paper", "approach", "method", "model", "results",
    "performance", "proposed", "data", "system", "learning", "algorithm", "network",
    "show", "our", "study", "analysis", "time", "application", "new", "used", "also",
    "more", "have", "has", "been", "such", "when", "where", "these", "into", "than",
    "over", "under", "between", "through", "after", "before", "during", "while",
    "high", "low", "large", "small", "good", "better", "best", "significant",
    "fig", "table", "eq", "start", "end", "all", "via", "make", "made", "work",
    "works", "research", "use", "uses", "user", "users", "case", "cases", "different",
    "experiments", "experimental", "however", "one", "two", "three", "first", "second",
    "task", "tasks", "state", "art", "out", "up", "down", "how", "what", "why",
    "problem", "problems", "solution", "solutions", "frame", "framework",
    "evaluation", "evaluate", "evaluated", "demonstrate", "shown", "showed", "present",
    "dataset", "datasets", "methodology", "introduction", "conclusion", "related",
    "future", "discussion", "findings", "novel", "existing", "compare", "comparison",
    "their", "they", "but", "not", "only", "its", "about", "other", "some", "most",
    "very", "many", "much", "well", "so", "up", "out", "if", "then", "else",
    "information", "process", "processing", "computing", "computer", "systems"
])

def load_data():
    """Loads all jsonl files matching the pattern."""
    files = glob.glob(os.path.join(DATA_DIR, "cs_2025_part_*.jsonl"))
    print(f"Found {len(files)} files.")
    
    all_papers = []
    for f_path in files:
        print(f"Loading {os.path.basename(f_path)}...")
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_papers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    df = pd.DataFrame(all_papers)
    print(f"Loaded {len(df)} records.")
    return df

def preprocess_data(df):
    """Preprocesses the dataframe."""
    # Date conversion
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['week_num'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Helper to extract primary institution type/country from authors
    def get_primary_inst_info(authors_list):
        if not authors_list:
            return "unknown", "unknown"
        # Heuristic: First author's primary affiliation
        first_auth = authors_list[0]
        return first_auth.get('inst_type', 'unknown'), first_auth.get('inst_country', 'unknown')

    df[['primary_inst_type', 'primary_inst_country']] = df['authors'].apply(
        lambda x: pd.Series(get_primary_inst_info(x))
    )

    return df

def save_plot(name, fig):
    """Saves plot to valid formats (JSON and PNG)."""
    # Save JSON for web embedding
    # Check if fig is plotly figure or matplotlib
    if hasattr(fig, 'write_json'):
        fig.write_json(os.path.join(OUTPUT_DIR, f"{name}.json"))
        try:
            fig.write_image(os.path.join(OUTPUT_DIR, f"{name}.png"), scale=2) 
            print(f"Saved {name} (json + png)")
        except Exception as e:
            print(f"Saved {name} (json only) - Error saving PNG: {e}")
    else:
        # Matplotlib figure (handled externally usually, but here for completeness)
        pass

# --- Analyses ---

def analyze_emerging_keywords(df):
    """
    Identifies emerging topics based on their growth velocity throughout the available 2025 data.
    """
    print("Analyzing Emerging Trends Data-Driven...")
    
    exploded = df[['week_num', 'all_topics']].explode('all_topics')
    exploded = exploded.dropna()
    exploded.rename(columns={'all_topics': 'Topic'}, inplace=True)
    
    weekly_total = df.groupby('week_num').size()
    topic_weekly_counts = exploded.groupby(['week_num', 'Topic']).size().reset_index(name='count')
    
    total_topic_counts = topic_weekly_counts.groupby('Topic')['count'].sum()
    significant_topics = total_topic_counts[total_topic_counts >= 50].index
    
    filtered_trends = topic_weekly_counts[topic_weekly_counts['Topic'].isin(significant_topics)].copy()
    
    filtered_trends['normalized_freq'] = filtered_trends.apply(
        lambda row: row['count'] / weekly_total.get(row['week_num'], 1), axis=1
    )
    
    import numpy as np
    
    topic_slopes = []
    
    for topic, group in filtered_trends.groupby('Topic'):
        if len(group) < 3: 
            continue
            
        min_week, max_week = df['week_num'].min(), df['week_num'].max()
        all_weeks = pd.DataFrame({'week_num': range(min_week, max_week + 1)})
        
        merged = pd.merge(all_weeks, group, on='week_num', how='left').fillna({'count': 0, 'normalized_freq': 0})
        
        x = merged['week_num']
        y = merged['normalized_freq']
        
        slope, intercept = np.polyfit(x, y, 1)
        mean_freq = y.mean()
        
        topic_slopes.append({
            'Topic': topic,
            'Velocity': slope,
            'MeanFreq': mean_freq,
            'TotalCount': group['count'].sum()
        })
        
    trends_df = pd.DataFrame(topic_slopes)
    
    top_growing = trends_df.sort_values('Velocity', ascending=False).head(20)
    
    fig = px.scatter(top_growing, x='TotalCount', y='Velocity', size='MeanFreq', text='Topic',
                     title="True Emerging Trends: Highest Growth Velocity in 2025",
                     labels={'Velocity': 'Growth Rate (Freq/Week)', 'TotalCount': 'Total Mentions'},
                     template="plotly_dark")
    fig.update_traces(textposition='top center')
    save_plot("emerging_keywords_velocity", fig)
    
    top_growing_table = top_growing[['Topic', 'Velocity', 'TotalCount']].to_dict('records')
    with open(os.path.join(OUTPUT_DIR, "top_emerging_trends.json"), 'w') as f:
        json.dump(top_growing_table, f, indent=2)

def analyze_institutional_dna(df):
    """Ratio of company vs education."""
    valid_types = df[~df['primary_inst_type'].isin(['unknown', None])]
    type_counts = valid_types['primary_inst_type'].value_counts().reset_index()
    type_counts.columns = ['Institution Type', 'Count']
    
    fig = px.pie(type_counts, values='Count', names='Institution Type', 
                 title="Institutional DNA (Primary Author Affiliation)", template="plotly_dark")
    save_plot("institutional_dna", fig)

def analyze_global_heatmaps(df):
    """Geographic clusters."""
    country_counts = df['primary_inst_country'].replace('unknown', float('nan')).dropna().value_counts().head(20).reset_index()
    country_counts.columns = ['Country Code', 'Paper Count']
    
    fig = px.bar(country_counts, x='Country Code', y='Paper Count', 
                 title="Top 20 Countries by Publication Volume", template="plotly_dark", color='Paper Count')
    save_plot("global_heatmaps", fig)

def analyze_freshness_index(df):
    """Citations per week relative to publication date."""
    current_week = df['week_num'].max()
    df['weeks_old'] = current_week - df['week_num'] + 1
    df['citation_velocity'] = df['citations'] / df['weeks_old']
    
    viral = df.sort_values('citation_velocity', ascending=False).head(100)
    
    fig = px.scatter(viral, x='week_num', y='citations', size='citation_velocity', color='citation_velocity',
                     hover_data=['title', 'date'],
                     title="Freshness Index: High Velocity Papers (Viral)",
                     labels={'week_num': "Publication Week", 'citations': "Total Citations", 'citation_velocity': "Citations/Week"},
                     template="plotly_dark")
    fig.add_annotation(x=viral['week_num'].max(), y=viral['citations'].max(), 
                       text="Recent & Highly Cited (Viral)", showarrow=True, arrowhead=1)
    
    save_plot("freshness_index", fig)

def analyze_subfield_momentum(df):
    """Subfield peaks over time."""
    top_subfields = df['subfield'].value_counts().head(5).index.tolist()
    filtered = df[df['subfield'].isin(top_subfields)]
    
    momentum = filtered.groupby(['quarter', 'subfield']).size().reset_index(name='Count')
    
    fig = px.bar(momentum, x="quarter", y="Count", color="subfield", barmode="group",
                 title="Subfield Momentum (Top 5)", template="plotly_dark")
    save_plot("subfield_momentum", fig)

def analyze_citation_long_tail(df):
    """Ref Count vs Citations to identify breakthroughs."""
    ref_median = df['ref_count'].median()
    cite_75 = df['citations'].quantile(0.75) 
    cite_95 = df['citations'].quantile(0.95)
    
    df['Category'] = 'Normal'
    
    breakthrough_mask = (df['ref_count'] < ref_median) & (df['citations'] > cite_95)
    df.loc[breakthrough_mask, 'Category'] = 'Potential Breakthrough'
    
    survey_mask = (df['ref_count'] > df['ref_count'].quantile(0.75)) & (df['citations'] > cite_95)
    df.loc[survey_mask, 'Category'] = 'High-Impact Survey'
    
    breakthroughs_df = df[breakthrough_mask][['id', 'title', 'citations', 'ref_count', 'date']]
    breakthroughs_df.to_json(os.path.join(OUTPUT_DIR, "breakthrough_candidates.json"), orient='records', indent=2)
    print(f"Saved {len(breakthroughs_df)} breakthrough candidates to breakthrough_candidates.json")

    plot_df = df.copy()
    if len(plot_df) > 5000:
        normals = plot_df[plot_df['Category'] == 'Normal'].sample(5000)
        others = plot_df[plot_df['Category'] != 'Normal']
        plot_df = pd.concat([others, normals])
        
    fig = px.scatter(plot_df, x='ref_count', y='citations', color='Category', hover_data=['title'],
                     title="The Citation Structure: Breakthroughs vs Surveys",
                     template="plotly_dark",
                     labels={'ref_count': "Reference Count (Breadth)", 'citations': "Citations (Impact)"},
                     color_discrete_map={'Normal': 'gray', 'Potential Breakthrough': 'red', 'High-Impact Survey': 'cyan'})
    
    save_plot("citation_long_tail", fig)

def analyze_abstract_keywords(df):
    """Abstract analysis: TF-IDF Keywords, Buzzwords, and WordCloud."""
    print("Analyzing Abstracts...")
    df['abstract_str'] = df['abstract'].fillna("").astype(str)
    
    # 1. Buzzword Tracker
    buzzwords = ["Agentic AI", "Generative", "Large Language Model", "Transformer", "Diffusion"]
    
    buzz_data = []
    weekly_volume = df.groupby('week_num').size()
    
    for kw in buzzwords:
        mask = df['abstract_str'].str.contains(kw, case=False, regex=False)
        weekly_counts = df[mask].groupby('week_num').size()
        
        for w in weekly_volume.index:
            count = weekly_counts.get(w, 0)
            total = weekly_volume.get(w, 1)
            fraction = count / total
            buzz_data.append({"Keyword": kw, "Week": w, "Fraction": fraction, "TotalPapers": total})
            
    if buzz_data:
        buzz_df = pd.DataFrame(buzz_data)
        fig1 = px.line(buzz_df, x="Week", y="Fraction", color="Keyword", markers=True,
                       title="Buzzword Trends (Normalized Frequency)", 
                       labels={'Fraction': "Proportion of Papers"},
                       template="plotly_dark")
        save_plot("buzzword_tracker", fig1)
    
    # 2. Advanced Keyword Extraction (TF-IDF)
    print("Running TF-IDF (Bigrams/Trigrams)...")
    tfidf = TfidfVectorizer(
        max_df=0.5, 
        min_df=50, 
        stop_words=list(STOP_WORDS),
        ngram_range=(2, 3), # Bigrams and Trigrams
        max_features=500
    )
    
    sample_size = min(30000, len(df))
    sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    try:
        tfidf_matrix = tfidf.fit_transform(sample_df['abstract_str'])
        feature_names = tfidf.get_feature_names_out()
        
        sum_scores = tfidf_matrix.sum(axis=0)
        
        tfidf_scores = []
        for col, term in enumerate(feature_names):
            tfidf_scores.append((term, sum_scores[0, col]))
            
        ranked_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:30]
        
        cw_df = pd.DataFrame(ranked_terms, columns=["Word", "TF-IDF Score"])
        
        fig2 = px.bar(cw_df, x="TF-IDF Score", y="Word", orientation='h', 
                      title=f"Top Distinctive Abstract Keywords (TF-IDF, n={sample_size})", template="plotly_dark")
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        save_plot("abstract_keywords_tfidf", fig2)
        
        # 3. WordCloud
        print("Generating WordCloud...")
        text = " ".join(sample_df['abstract_str'])
        
        wc = WordCloud(
            width=800, height=400, 
            background_color='black', 
            stopwords=STOP_WORDS,
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Save WordCloud 
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title("Abstract Word Cloud")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "abstract_wordcloud.png"))
        plt.close()
        print("Saved abstract_wordcloud.png")
        
    except Exception as e:
        print(f"Error in Keyword Analysis: {e}")

def main():
    print("Starting Analysis...")
    df = load_data()
    
    if df.empty:
        print("No data found. Exiting.")
        return

    print("Preprocessing...")
    df = preprocess_data(df)
    
    print("Running Analyses...")
    analyze_emerging_keywords(df)
    analyze_institutional_dna(df)
    analyze_global_heatmaps(df)
    analyze_freshness_index(df)
    analyze_subfield_momentum(df)
    analyze_citation_long_tail(df)
    analyze_abstract_keywords(df)
    
    print(f"All analyses completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
