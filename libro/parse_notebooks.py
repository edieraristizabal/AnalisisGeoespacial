import json
import glob
import re

files = glob.glob('*.ipynb') + glob.glob('*.Rmd') + glob.glob('*.md')

citation_pattern = re.compile(r'\{cite\s*`?[^}`]+`?\}|\[@[^\]]+\]')

topics_missing_citations = []
citations_found = set()

for f in files:
    if f in ['requirements.txt', 'how_update.txt']: continue
    try:
        if f.endswith('.ipynb'):
            with open(f, 'r', encoding='utf-8') as file:
                nb = json.load(file)
            cells = [c['source'] for c in nb.get('cells', []) if c['cell_type'] == 'markdown']
            markdown_text = []
            for c in cells:
                if isinstance(c, list):
                    markdown_text.extend(c)
                else:
                    markdown_text.append(c)
            full_text = ''.join(markdown_text)
        else:
            with open(f, 'r', encoding='utf-8') as file:
                full_text = file.read()
        
        # Find all citations
        found = citation_pattern.findall(full_text)
        citations_found.update(found)
        
        # Split by headers
        sections = re.split(r'\n#+\s+', full_text)
        for i, sec in enumerate(sections):
            if i == 0: continue # Skip before first header
            lines = sec.split('\n')
            header = lines[0].strip()
            content = '\n'.join(lines[1:]).strip()
            
            # Simple heuristic: if a section has more than 300 characters of text 
            # and no citations, it might need one.
            if len(content) > 300 and not citation_pattern.search(content):
                # Check if it looks like theoretical text (has words, not just code blocks)
                text_only = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
                if len(text_only.split()) > 50:
                    topics_missing_citations.append((f, header))
                
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("CITATION FORMATS FOUND:")
for c in list(citations_found)[:10]:
    print(c)
print("\nTOPICS MISSING CITATIONS (Sample):")
for t in topics_missing_citations[:20]:
    print(f"{t[0]}: {t[1]}")
