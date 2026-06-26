import fitz
doc = fitz.open('ncert_history.pdf')
for i in range(15, 25):
    page = doc.load_page(i)
    blocks = page.get_text('dict').get('blocks', [])
    for b in blocks:
        if 'lines' in b:
            for l in b['lines']:
                for s in l['spans']:
                    text = s['text'].strip()
                    if text and len(text) > 3:
                        print(f"Page {i+1} | Size {s.get('size', 0):.1f} | Font {s.get('font', '')} | {text}")
