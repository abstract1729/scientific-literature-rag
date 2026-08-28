from src.colpali import CorpusRenderer


PDF_DIR = "data/raw/papers"

renderer = CorpusRenderer(pdf_dir=PDF_DIR,dpi=150)
renderer.render_corpus(overwrite=False)