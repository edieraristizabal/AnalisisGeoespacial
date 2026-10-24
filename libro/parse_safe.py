import json
import glob
import re

files = glob.glob('*.ipynb') + glob.glob('*.Rmd') + glob.glob('*.md')

topics = {
    'Kriging': r'\bkriging\b',
    "Moran's I": r'\bmoran\b',
    'LISA': r'\blisa\b',
    'GWR': r'\bgwr\b',
    'SAR': r'\bsar\b',
    'CAR': r'\bcar\b',
    'Point Patterns': r'procesos puntuales|patrones de puntos',
    'Gaussian Processes': r'procesos gaussianos'
}

for f in files:
    try:
        content = ""
        if f.endswith('.ipynb'):
            with open(f, 'r', encoding='utf-8') as file:
                nb = json.load(file)
            cells = [c['source'] for c in nb.get('cells', []) if c['cell_type'] == 'markdown']
            content = ''.join([s if isinstance(s, str) else ''.join(s) for s in cells])
        else:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                
        content = content.lower()
        
        has_cite = '{cite' in content or '[@' in content
        
        found_topics = []
        for t, pat in topics.items():
            if re.search(pat, content):
                found_topics.append(t)
                
        if found_topics:
            print(f"{f} mentions: {', '.join(found_topics)}. Has citations: {has_cite}")
            
    except Exception as e:
        print(f"Error {f}: {e}")

