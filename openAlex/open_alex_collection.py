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

    selected_fields = (
        "id,doi,title,publication_date,primary_location,authorships,abstract_inverted_index,"
        "primary_topic,topics,cited_by_count,referenced_works,open_access"
    )
    params = {
        'filter': 'publication_year:2025,primary_topic.field.id:fields/17',
        'select': selected_fields,
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
                    # 1. Extract and Deduplicate Geography
                    # authorships.countries is a list of ISO codes like ["US", "DE"]
                    countries = list(set(work.get('countries', []))) 

                    # 2. Extract Authors with Institution Metadata
                    authors_data = []
                    for auth in work.get('authorships', []):
                        # Get the primary institution details (OpenAlex usually puts the main one first)
                        insts = auth.get('institutions', [])
                        primary_inst = insts[0] if insts else {}
                        
                        authors_data.append({
                            'name': safe_get(auth, 'author', 'display_name', default="Unknown"),
                            'id': safe_get(auth, 'author', 'id'),
                            'institution': primary_inst.get('display_name', "No Affiliation"),
                            'inst_type': primary_inst.get('type', "unknown"), # e.g., 'company', 'education', 'government'
                            'inst_country': primary_inst.get('country_code', "unknown")
                        })

                    # 3. Final Paper Object Construction
                    paper = {
                        'id': work.get('id'),
                        'doi': work.get('doi'),
                        'title': work.get('title'),
                        'date': work.get('publication_date'),
                        'citations': work.get('cited_by_count', 0),
                        
                        # VENUE & ORIGIN
                        'venue': safe_get(work, 'primary_location', 'source', 'display_name', default="Unknown Venue"),
                        'venue_type': safe_get(work, 'primary_location', 'source', 'type', default="unknown"), # journal, conference, preprint
                        
                        # TOPIC ANALYSIS (The "Hot" Indicators)
                        'primary_topic': safe_get(work, 'primary_topic', 'display_name', default="Unknown Topic"),
                        'subfield': safe_get(work, 'primary_topic', 'subfield', 'display_name', default="Unknown Subfield"),
                        'all_topics': [t.get('display_name') for t in work.get('topics', [])], # Full topic cluster
                        
                        # TREND METRICS
                        'abstract': rebuild_abstract(work.get('abstract_inverted_index')),
                        'authors': authors_data,
                        'countries': countries,
                        'is_oa': safe_get(work, 'open_access', 'is_oa', default=False),
                        'ref_count': len(work.get('referenced_works', [])), # Breadth of the literature review
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

if __name__ == "__main__":
    fetch_2025_cs("chirag.m@nyu.edu", records_per_file=10000)
