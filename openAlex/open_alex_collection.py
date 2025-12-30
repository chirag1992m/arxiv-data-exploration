# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "requests>=2.32.5",
# ]
# ///

import requests
import json
import os
import time

def rebuild_abstract(inverted_index):
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    try:
        word_index = []
        for word, locations in inverted_index.items():
            for loc in locations: word_index.append((loc, word))
        word_index.sort()
        return " ".join(word for _, word in word_index)
    except Exception: return None

def safe_get(data, *keys, default=None):
    """Safely traverses nested dictionaries."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else: return default
    return data if data is not None else default

def fetch_2025_cs(email, base_name='cs_2025', records_per_file=50000):
    state_file = 'state.json'
    cursor = '*'
    total_collected = 0
    total_available = 0 # Initialize total
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            cursor = state.get('cursor', '*')
            total_collected = state.get('count', 0)
        print(f"Resuming... Already have {total_collected} records.")

    file_num = (total_collected // records_per_file) + 1
    current_filename = f"{base_name}_part_{file_num}.jsonl"
    f_handle = open(current_filename, 'a', encoding='utf-8')

    params = {
        'filter': 'publication_year:2025,primary_topic.field.id:fields/17',
        'select': 'id,doi,title,publication_date,primary_location,authorships,primary_topic,abstract_inverted_index,cited_by_count',
        'per-page': 200, 
        'cursor': cursor,
        'mailto': email
    }

    try:
        while params['cursor']:
            try:
                response = requests.get("https://api.openalex.org/works", params=params, timeout=30)
                if response.status_code == 429:
                    time.sleep(60); continue
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                if not results: break

                # CAPTURE TOTAL ON FIRST SUCCESSFUL CALL
                if total_available == 0:
                    total_available = data.get('meta', {}).get('count', 0)

                for work in results:
                    # [Author extraction logic here...]
                    authors_data = [{'name': safe_get(a, 'author', 'display_name'), 
                                     'affiliation': (a.get('institutions') or [{}])[0].get('display_name')} 
                                    for a in work.get('authorships', [])]

                    paper = {
                        'title': work.get('title'),
                        'citations': work.get('cited_by_count', 0),
                        'authors': authors_data,
                        # ... other fields
                    }
                    
                    f_handle.write(json.dumps(paper) + '\n')
                    total_collected += 1

                    # Rotation logic...
                    if total_collected % records_per_file == 0:
                        f_handle.close()
                        file_num += 1
                        current_filename = f"{base_name}_part_{file_num}.jsonl"
                        f_handle = open(current_filename, 'a', encoding='utf-8')

                # UPDATE PROGRESS BAR
                # We use end='\r' and flush=True to make it live
                # We add '    ' at the end to clear any leftover characters
                print(f"Collected: {total_collected:,} / {total_available:,} | "
                      f"Latest: {paper['title'][:30]}...    ", 
                      end='\r', flush=True)

                f_handle.flush()
                os.fsync(f_handle.fileno())
                params['cursor'] = data.get('meta', {}).get('next_cursor')
                
                with open(state_file, 'w') as sf:
                    json.dump({'cursor': params['cursor'], 'count': total_collected}, sf)

            except Exception as e:
                print(f"\nError: {e}. Retrying...")
                time.sleep(10); continue

    finally:
        f_handle.close()
        print(f"\n\nDone! Final count: {total_collected}")

# Start
fetch_2025_cs("chirag.m@nyu.edu", records_per_file=5000)
