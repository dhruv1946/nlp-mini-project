from collections import Counter
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag as nltk_pos_tag
import importlib
import nlp_engine
importlib.reload(nlp_engine)
from nlp_engine import Preprocessor, NGramAnalyzer, HMMPosTagger, CYKParser, ChunkExtractor, NERExtractor, PlagiarismDetector, DocumentParser
st.set_page_config(page_title='Content Evaluator // NLP Pipeline', page_icon='▣', layout='wide', initial_sidebar_state='collapsed')
_CSS = '\n<style>\n@import url(\'https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap\');\n\n*, *::before, *::after {\n    box-sizing: border-box;\n}\n\nhtml, body, .stApp, [data-testid="stAppViewContainer"],\n[data-testid="stHeader"], section[data-testid="stSidebar"] {\n    background-color: #000000 !important;\n    color: #FFFFFF !important;\n    font-family: \'Syne\', sans-serif !important;\n}\n\nh1, h2, h3, h4, h5, h6,\n[data-testid="stMarkdownContainer"] h1,\n[data-testid="stMarkdownContainer"] h2,\n[data-testid="stMarkdownContainer"] h3 {\n    color: #FFFFFF !important;\n    letter-spacing: 0.04em;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n#MainMenu, footer, header { visibility: hidden; }\n\n.block-container {\n    padding: 2.5rem 3rem 4rem !important;\n    max-width: 1280px !important;\n}\n\n.hero-title {\n    font-size: clamp(2rem, 4vw, 3.2rem);\n    font-weight: 800;\n    letter-spacing: -0.02em;\n    line-height: 1.05;\n    border-bottom: 1px solid #FFFFFF;\n    padding-bottom: 0.9rem;\n    margin-bottom: 0.3rem;\n    font-family: \'Syne\', sans-serif !important;\n}\n.hero-sub {\n    font-size: 0.72rem;\n    color: #666666;\n    letter-spacing: 0.22em;\n    text-transform: uppercase;\n    margin-bottom: 2.5rem;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n.phase-eyebrow {\n    font-size: 0.65rem;\n    letter-spacing: 0.22em;\n    text-transform: uppercase;\n    color: #555555;\n    margin-bottom: 0.2rem;\n    font-family: \'Syne\', sans-serif !important;\n}\n.phase-title {\n    font-size: 1.3rem;\n    font-weight: 700;\n    border-left: 2px solid #FFFFFF;\n    padding-left: 0.7rem;\n    margin-bottom: 1.4rem;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n.metric-card {\n    border: 1px solid #222222;\n    padding: 1rem 1.2rem;\n    background: #080808;\n    height: 100%;\n}\n.metric-label {\n    font-size: 0.6rem;\n    letter-spacing: 0.18em;\n    text-transform: uppercase;\n    color: #555555;\n    margin-bottom: 0.2rem;\n    font-family: \'Syne\', sans-serif !important;\n}\n.metric-value {\n    font-size: 1.9rem;\n    font-weight: 700;\n    color: #FFFFFF;\n    line-height: 1.1;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n.tag-badge {\n    display: inline-block;\n    border: 1px solid #2a2a2a;\n    padding: 0.1rem 0.45rem;\n    font-size: 0.72rem;\n    margin: 0.1rem 0.15rem;\n    background: #0d0d0d;\n    color: #cccccc;\n    font-family: \'Syne\', monospace !important;\n}\n\n.source-badge {\n    display: inline-block;\n    padding: 0.12rem 0.45rem;\n    font-size: 0.62rem;\n    text-transform: uppercase;\n    letter-spacing: 0.1em;\n    border: 1px solid #333333;\n    margin-bottom: 0.4rem;\n    font-family: \'Syne\', monospace !important;\n}\n.source-web    { border-color: #0070F3 !important; color: #58A6FF !important; }\n.source-corpus { border-color: #555555 !important; color: #888888 !important; }\n.source-upload { border-color: #30D158 !important; color: #30D158 !important; }\n.source-academic { border-color: #FF9F0A !important; color: #FFD60A !important; }\n\n.risk-high    { color: #FF453A !important; }\n.risk-medium  { color: #FF9F0A !important; }\n.risk-low     { color: #30D158 !important; }\n.risk-minimal { color: #555555 !important; }\n\n.formula-box {\n    border-left: 2px solid #333333;\n    padding: 0.45rem 0.9rem;\n    background: #0a0a0a;\n    font-size: 0.78rem;\n    color: #888888;\n    margin: 0.6rem 0 1rem;\n    letter-spacing: 0.03em;\n    font-family: monospace !important;\n}\n.info-box {\n    border: 1px solid #1a1a1a;\n    padding: 0.65rem 0.9rem;\n    background: #080808;\n    font-size: 0.78rem;\n    color: #666666;\n    margin-bottom: 1rem;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n.sep { border: none; border-top: 1px solid #141414; margin: 2rem 0; }\n\n.stTextArea textarea {\n    background: #070707 !important;\n    color: #FFFFFF !important;\n    border: 1px solid #282828 !important;\n    border-radius: 0 !important;\n    font-size: 0.88rem !important;\n    font-family: \'Syne\', monospace !important;\n    resize: vertical !important;\n}\n.stTextArea textarea:focus {\n    border-color: #FFFFFF !important;\n    box-shadow: none !important;\n}\n.stTextArea label { color: #555555 !important; font-size: 0.7rem !important; font-family: \'Syne\', sans-serif !important; }\n\n.stButton > button {\n    background: #FFFFFF !important;\n    color: #000000 !important;\n    border: none !important;\n    border-radius: 0 !important;\n    font-weight: 700 !important;\n    letter-spacing: 0.12em !important;\n    text-transform: uppercase !important;\n    padding: 0.55rem 1.8rem !important;\n    font-family: \'Syne\', sans-serif !important;\n    transition: background 0.12s !important;\n}\n.stButton > button:hover { background: #CCCCCC !important; }\n\n[data-testid="stFileUploader"] {\n    border: 1px dashed #282828 !important;\n    background: #070707 !important;\n    border-radius: 0 !important;\n    padding: 0.75rem !important;\n}\n[data-testid="stFileUploader"] section {\n    background: transparent !important;\n    padding: 0 !important;\n}\n[data-testid="stFileUploader"] label { color: #555555 !important; font-size: 0.75rem !important; font-family: \'Syne\', sans-serif !important; }\n[data-testid="stFileUploader"] button {\n    background: #111111 !important;\n    color: #FFFFFF !important;\n    border: 1px solid #333333 !important;\n    border-radius: 0 !important;\n    font-size: 0.75rem !important;\n    letter-spacing: 0.08em !important;\n    padding: 0.4rem 1.2rem !important;\n    font-family: \'Syne\', sans-serif !important;\n}\n[data-testid="stFileUploader"] button:hover {\n    background: #222222 !important;\n    border-color: #666666 !important;\n}\n[data-testid="stFileUploader"] small {\n    color: #555555 !important;\n    font-size: 0.68rem !important;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n[data-testid="stExpander"] {\n    border: 1px solid #1e1e1e !important;\n    border-radius: 0 !important;\n    background: #050505 !important;\n    margin-bottom: 0.6rem !important;\n}\n[data-testid="stExpander"] summary {\n    color: #AAAAAA !important;\n    padding: 0.7rem 1rem !important;\n    background: #0a0a0a !important;\n}\n[data-testid="stExpander"] summary:hover { color: #FFFFFF !important; }\n[data-testid="stExpander"] summary p,\n[data-testid="stExpander"] summary span:not([data-testid*="Icon"]):not([class*="material"]) {\n    font-size: 0.78rem !important;\n    letter-spacing: 0.12em !important;\n    text-transform: uppercase !important;\n    font-family: \'Syne\', sans-serif !important;\n}\n\n[data-testid*="Icon"],\n[data-testid*="icon"],\n[data-testid="stExpanderToggleIcon"],\n.material-symbols-rounded,\n.material-symbols-outlined,\n.material-icons,\nspan[class*="material-symbols"],\nspan[class*="material-icons"],\ni[class*="material-icons"],\ni[class*="material-symbols"] {\n    font-family: \'Material Symbols Rounded\', \'Material Icons\', sans-serif !important;\n    font-weight: normal !important;\n    font-style: normal !important;\n    text-transform: none !important;\n    letter-spacing: normal !important;\n    line-height: 1 !important;\n    display: inline-block !important;\n    white-space: nowrap !important;\n    word-wrap: normal !important;\n    direction: ltr !important;\n    -webkit-font-smoothing: antialiased !important;\n    text-rendering: optimizeLegibility !important;\n    font-feature-settings: \'liga\' !important;\n}\n\n[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid #1e1e1e !important; }\n[data-testid="stTabs"] button {\n    color: #555555 !important;\n    font-size: 0.72rem !important;\n    letter-spacing: 0.12em !important;\n    text-transform: uppercase !important;\n    border-radius: 0 !important;\n    background: transparent !important;\n    font-family: \'Syne\', sans-serif !important;\n}\n[data-testid="stTabs"] button[aria-selected="true"] {\n    color: #FFFFFF !important;\n    border-bottom: 2px solid #FFFFFF !important;\n    font-weight: 700 !important;\n}\n\n[data-testid="stDataFrame"] {\n    border: 1px solid #1a1a1a !important;\n}\n[data-testid="stDataFrame"] th {\n    background: #0d0d0d !important;\n    color: #555555 !important;\n    font-size: 0.62rem !important;\n    letter-spacing: 0.12em !important;\n    text-transform: uppercase !important;\n    border-bottom: 1px solid #222222 !important;\n    font-family: \'Syne\', sans-serif !important;\n}\n[data-testid="stDataFrame"] td {\n    color: #DDDDDD !important;\n    background: #060606 !important;\n    border-bottom: 1px solid #111111 !important;\n    font-size: 0.82rem !important;\n    font-family: \'Syne\', monospace !important;\n}\n\n.stProgress > div > div { background: #FFFFFF !important; }\n\n.stAlert { border-radius: 0 !important; border: 1px solid #222222 !important; background: #080808 !important; }\n\n::-webkit-scrollbar { width: 4px; height: 4px; }\n::-webkit-scrollbar-track { background: #000; }\n::-webkit-scrollbar-thumb { background: #2a2a2a; }\n\n.displacy-wrapper {\n    background: #080808;\n    border: 1px solid #1e1e1e;\n    padding: 1rem 1.2rem;\n    line-height: 2.8;\n    font-size: 0.88rem;\n    color: #DDDDDD;\n    overflow-x: auto;\n}\nmark.entity { font-weight: 600; border-radius: 0 !important; }\nspan.label  { font-size: 0.6rem !important; letter-spacing: 0.08em; }\n</style>\n'
st.markdown(_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models():
    return (Preprocessor(), HMMPosTagger(), NERExtractor(), CYKParser(), ChunkExtractor())

def _badge(text: str) -> str:
    return f'<span class="tag-badge">{text}</span>'

def _metric(label: str, value) -> str:
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'

def _phase(num: str, title: str):
    st.markdown(f'<hr class="sep"><div class="phase-eyebrow">Phase {num}</div><div class="phase-title">{title}</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">CONTENT EVALUATOR</div><div class="hero-sub">Automated Quality &amp; Plagiarism Analysis &nbsp;//&nbsp; Multi-Stage NLP Pipeline</div>', unsafe_allow_html=True)
with st.spinner('Initialising NLP models — HMM training on Brown corpus…'):
    preprocessor, hmm_tagger, ner_extractor, cyk_parser, chunk_extractor = load_models()
st.markdown('### ▸ INPUT DOCUMENT')
input_doc_file = st.file_uploader(label='Upload document (.pdf, .docx, .txt)', type=['pdf', 'docx', 'doc', 'txt'], key='main_doc_upload')
if input_doc_file is not None:
    if st.session_state.get('_last_uploaded_doc') != input_doc_file.name:
        st.session_state['_last_uploaded_doc'] = input_doc_file.name
        extracted_doc_text = DocumentParser.extract_text(input_doc_file, input_doc_file.name)
        st.session_state['main_input'] = extracted_doc_text
        if extracted_doc_text.strip():
            st.success(f'Loaded {len(extracted_doc_text.split())} words from {input_doc_file.name}')
        else:
            st.warning('No readable text could be extracted from this file. (Note: scanned image-only PDFs do not contain text layers).')
elif '_last_uploaded_doc' in st.session_state and input_doc_file is None:
    del st.session_state['_last_uploaded_doc']
input_text = st.text_area(label='Document text (pasted or extracted from uploaded file)', placeholder='Paste document text here, or upload a .pdf, .docx, or .txt file above…', height=220, key='main_input', label_visibility='visible')
if not input_text.strip() and input_doc_file is not None:
    input_text = DocumentParser.extract_text(input_doc_file, input_doc_file.name)
st.markdown('<div class="info-box">Autonomous Plagiarism Engine cross-references Open Academic APIs (IEEE, Springer, Wiley, Crossref), Live Web, Live Wikipedia, and the Academic Benchmark Corpus.</div>', unsafe_allow_html=True)
col_btn, _ = st.columns([1, 6])
with col_btn:
    run_btn = st.button('▶  ANALYSE DOCUMENT', key='run_analysis')
with st.expander('▾ OPTIONAL: Offline Reference Files (.pdf, .docx, .txt)'):
    offline_files = st.file_uploader(label='Compare against specific offline documents', type=['pdf', 'docx', 'doc', 'txt'], accept_multiple_files=True, key='offline_ref_upload')
if not run_btn:
    st.stop()
if not input_text.strip():
    st.warning('⚠  Please enter some text before running the analysis.')
    st.stop()
_phase('01', 'PREPROCESSING PIPELINE')
with st.spinner('Tokenising and lemmatising…'):
    prep = preprocessor.preprocess(input_text)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(_metric('Sentences', prep['num_sentences']), unsafe_allow_html=True)
c2.markdown(_metric('Raw Tokens', prep['num_tokens']), unsafe_allow_html=True)
c3.markdown(_metric('After Stop-word Filter', prep['num_filtered']), unsafe_allow_html=True)
c4.markdown(_metric('Vocabulary Size', prep['vocab_size']), unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)
with st.expander('TOKENISATION & LEMMATISATION DETAILS'):
    col_s, col_l = st.columns(2)
    with col_s:
        st.markdown('**Detected Sentences**')
        for i, s in enumerate(prep['sentences'], 1):
            display = s[:90] + ('…' if len(s) > 90 else '')
            st.markdown(_badge(f'{i}.&nbsp;{display}'), unsafe_allow_html=True)
    with col_l:
        st.markdown('**Lemmatised Tokens (first 50)**')
        sample_html = ''.join((_badge(l) for l in prep['lemmas'][:50]))
        st.markdown(sample_html or '*No tokens after filtering.*', unsafe_allow_html=True)
_phase('02', 'FLUENCY & SYNTAX EVALUATION')
with st.expander('N-GRAM FREQUENCY ANALYSIS  +  LAPLACE SMOOTHING'):
    with st.spinner('Computing n-grams…'):
        ngram_analyzer = NGramAnalyzer(prep['filtered_tokens'])
        stats = ngram_analyzer.get_all_stats()
    st.markdown('<div class="formula-box">Laplace:&nbsp;&nbsp;P(w₂ | w₁) = [C(w₁,w₂) + 1] / [C(w₁) + |V|]<br>Log P(sentence) = Σ log P_laplace(wᵢ | wᵢ₋₁)</div>', unsafe_allow_html=True)
    t_uni, t_bi, t_tri, t_sent = st.tabs(['UNIGRAMS', 'BIGRAMS', 'TRIGRAMS', 'SENTENCE LOG-PROB'])
    with t_uni:
        df = pd.DataFrame(stats['unigrams'], columns=['Token', 'Frequency'])
        st.dataframe(df, width='stretch', hide_index=True)
    with t_bi:
        rows = [(' '.join(g), c) for g, c in stats['bigrams']]
        df = pd.DataFrame(rows, columns=['Bigram', 'Frequency'])
        st.dataframe(df, width='stretch', hide_index=True)
        st.markdown('**Laplace Probabilities — top bigrams**')
        lp_rows = []
        for (w1, w2), cnt in stats['bigrams'][:8]:
            p = ngram_analyzer.laplace_bigram_prob(w1, w2)
            lp_rows.append({'Bigram': f'{w1} → {w2}', 'Count': cnt, 'P_laplace': f'{p:.7f}'})
        st.dataframe(pd.DataFrame(lp_rows), width='stretch', hide_index=True)
    with t_tri:
        rows = [(' '.join(g), c) for g, c in stats['trigrams']]
        df = pd.DataFrame(rows, columns=['Trigram', 'Frequency'])
        st.dataframe(df, width='stretch', hide_index=True)
    with t_sent:
        sent_rows = []
        for s in prep['sentences'][:8]:
            toks = word_tokenize(s)
            lp = ngram_analyzer.sentence_log_prob(toks)
            sent_rows.append({'Sentence': s[:70] + '…' if len(s) > 70 else s, 'Log P (bigram LM)': f'{lp:.4f}'})
        st.dataframe(pd.DataFrame(sent_rows), width='stretch', hide_index=True)
        st.markdown('<div class="info-box">More negative → sequence is less expected by the bigram language model.<br>Unusually low scores may indicate awkward / non-native phrasing.</div>', unsafe_allow_html=True)
with st.expander('HMM POS TAGGER  —  VITERBI ALGORITHM'):
    first_sent = prep['sentences'][0] if prep['sentences'] else input_text[:200]
    with st.spinner('Viterbi decoding…'):
        hmm_tagged = hmm_tagger.tag(first_sent)
        nltk_tagged = nltk_pos_tag([w for w, _ in hmm_tagged], tagset='universal')
    st.markdown('<div class="formula-box">δₜ(s) = max_{s\'} [ δₜ₋₁(s\') × A(s\'→s) × B(s, wₜ) ]&nbsp;&nbsp;&nbsp;ψₜ(s) = argmax_{s\'} [ δₜ₋₁(s\') × A(s\'→s) ]</div>', unsafe_allow_html=True)
    st.caption(f"Analysing: *{first_sent[:100]}{('…' if len(first_sent) > 100 else '')}*")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('**HMM Viterbi Tags** *(Brown corpus)*')
        st.dataframe(pd.DataFrame(hmm_tagged, columns=['Word', 'HMM Tag']), width='stretch', hide_index=True)
    with col_b:
        st.markdown('**NLTK Reference Tags** *(Perceptron, universal)*')
        st.dataframe(pd.DataFrame(nltk_tagged, columns=['Word', 'NLTK Tag']), width='stretch', hide_index=True)
    st.markdown('**POS Distribution — full document**')
    all_alpha = [t for t in prep['raw_tokens'] if t.isalpha()]
    all_tagged = nltk_pos_tag(all_alpha, tagset='universal')
    tag_counts = Counter((tag for _, tag in all_tagged))
    df_dist = pd.DataFrame(tag_counts.most_common(), columns=['POS Tag', 'Count'])
    st.dataframe(df_dist, width='stretch', hide_index=True)
with st.expander('CYK PARSER  —  SYNTAX VALIDITY CHECK'):
    st.markdown('<div class="formula-box">table[i][j] = { A | A→BC, B∈table[i][k], C∈table[k+1][j] }&nbsp;&nbsp;(∀ i ≤ k &lt; j)<br>Valid  ⟺  S ∈ table[0][n-1]</div>', unsafe_allow_html=True)
    cyk_rows = []
    for sent in prep['sentences'][:6]:
        toks = [t for t in word_tokenize(sent) if t.isalpha()]
        if not toks:
            continue
        tagged = nltk_pos_tag(toks)
        pos_seq = [tag for _, tag in tagged]
        result = cyk_parser.parse(pos_seq)
        validity = '✓  VALID' if result['is_valid'] else '✗  INVALID'
        cyk_rows.append({'Sentence': sent[:60] + '…' if len(sent) > 60 else sent, 'POS Sequence': ' '.join(pos_seq[:10]) + ('…' if len(pos_seq) > 10 else ''), 'CNF Result': validity})
    st.dataframe(pd.DataFrame(cyk_rows), width='stretch', hide_index=True)
    st.markdown('<div class="info-box">Grammar covers: S→NP VP | NP ADJP, NP→DT NN|NNS|NNP | JJ NN | PRP …, VP→VB|VBZ NP | MD VP …, PP→IN NP. Sentences not derivable from S are flagged INVALID.</div>', unsafe_allow_html=True)
_phase('03', 'SEMANTIC EXTRACTION')
with st.expander('REGEX CHUNKING  —  NOUN PHRASES & VERB PHRASES'):
    with st.spinner('Extracting NP/VP chunks…'):
        chunks = chunk_extractor.extract(nltk_pos_tag(all_alpha))
    col_np, col_vp = st.columns(2)
    with col_np:
        nps = chunks['noun_phrases']
        st.markdown(f'**Noun Phrases** &nbsp;·&nbsp; *{len(nps)} found*')
        html = ''.join((_badge(p) for p in nps[:30]))
        st.markdown(html or '*None detected.*', unsafe_allow_html=True)
    with col_vp:
        vps = chunks['verb_phrases']
        st.markdown(f'**Verb Phrases** &nbsp;·&nbsp; *{len(vps)} found*')
        html = ''.join((_badge(p) for p in vps[:30]))
        st.markdown(html or '*None detected.*', unsafe_allow_html=True)
with st.expander('NAMED ENTITY RECOGNITION  —  spaCy  en_core_web_sm'):
    with st.spinner('Running NER pipeline…'):
        ner_result = ner_extractor.extract(input_text)
    st.markdown(f"**{ner_result['entity_count']} entit{('ies' if ner_result['entity_count'] != 1 else 'y')} detected**")
    if ner_result['entity_count'] > 0:
        wrapped_html = '<div class="displacy-wrapper">' + ner_result['html'] + '</div>'
        st.markdown(wrapped_html, unsafe_allow_html=True)
        st.markdown('<br>**Entity Breakdown**', unsafe_allow_html=True)
        df_ner = pd.DataFrame(ner_result['entities'])[['text', 'label', 'description']].rename(columns={'text': 'Entity', 'label': 'Type', 'description': 'Description'})
        st.dataframe(df_ner, width='stretch', hide_index=True)
    else:
        st.markdown('<div class="info-box">No named entities found in the input text.</div>', unsafe_allow_html=True)
_phase('04', 'HYBRID PLAGIARISM DETECTION')
st.markdown('<div class="formula-box">Combined = 0.40 × TF-IDF + 0.30 × BoW + 0.30 × Token Coverage &nbsp;|&nbsp; High ≥ 35% &nbsp;|&nbsp; Medium ≥ 20% &nbsp;|&nbsp; Low ≥ 10%</div>', unsafe_allow_html=True)
with st.spinner('Cross-referencing Academic Repositories (IEEE/Springer/Crossref), Live Web, and Wikipedia…'):
    detector = PlagiarismDetector()
    fingerprint_queries = detector.extract_search_queries(input_text, entities=ner_result.get('entities'), max_queries=8)
    plag_result = detector.analyze_hybrid(input_text, uploaded_files=offline_files, search_terms=fingerprint_queries, corpus_dir='corpus')
if plag_result['error']:
    st.error(plag_result['error'])
elif not plag_result['results']:
    st.markdown('<div class="info-box">No reference sources matched or available.</div>', unsafe_allow_html=True)
else:
    top_match = plag_result['results'][0]
    if top_match['risk_level'] in ('HIGH', 'MEDIUM'):
        st.markdown(f"""<div class="info-box" style="border-left: 3px solid #FF453A; background: #120505; color: #FF453A !important; font-size: 0.88rem; font-weight: 600;">⚠ Plagiarism Alert: {top_match['risk_level']} similarity detected with {top_match['reference']} ({top_match['combined']}%)</div>""", unsafe_allow_html=True)
    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    c_s1.markdown(_metric('Sources Scanned', plag_result['sources_count']), unsafe_allow_html=True)
    c_s2.markdown(_metric('Peak Combined Sim', f"{top_match['combined']}%"), unsafe_allow_html=True)
    c_s3.markdown(f"""<div class="metric-card"><div class="metric-label">Highest Risk</div><div class="metric-value risk-{top_match['risk_level'].lower()}">{top_match['risk_level']}</div></div>""", unsafe_allow_html=True)
    c_s4.markdown(_metric('Top Match Source', top_match['source_type']), unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('**Comparative Similarity Breakdown**')
    for res in plag_result['results'][:6]:
        risk_cls = 'risk-' + res['risk_level'].lower()
        stype = res['source_type']
        badge_cls = 'source-academic' if 'Academic' in stype else 'source-upload' if 'Upload' in stype else 'source-corpus' if 'Corpus' in stype else 'source-web'
        c_ref, c_tfidf, c_bow, c_cov, c_comb, c_risk = st.columns([3.2, 1.1, 1.1, 1.1, 1.1, 1.1])
        c_ref.markdown(f"""<div class="metric-card"><div class="source-badge {badge_cls}">{stype}</div><div style="font-size:0.92rem;font-weight:600;color:#CCCCCC;">{res['reference']}</div></div>""", unsafe_allow_html=True)
        c_tfidf.markdown(_metric('TF-IDF', f"{res['tfidf_score']}%"), unsafe_allow_html=True)
        c_bow.markdown(_metric('BoW Sim', f"{res['bow_score']}%"), unsafe_allow_html=True)
        c_cov.markdown(_metric('Coverage', f"{res['coverage_score']}%"), unsafe_allow_html=True)
        c_comb.markdown(_metric('Combined', f"{res['combined']}%"), unsafe_allow_html=True)
        c_risk.markdown(f"""<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{res['risk_level']}</div></div>""", unsafe_allow_html=True)
        st.progress(min(int(res['combined']), 100))
        st.markdown('<br>', unsafe_allow_html=True)
    with st.expander('FULL HYBRID SIMILARITY MATRIX'):
        df_plag = pd.DataFrame(plag_result['results'])
        df_plag.columns = [c.replace('_', ' ').title() for c in df_plag.columns]
        st.dataframe(df_plag, width='stretch', hide_index=True)
        st.markdown('<div class="info-box">Combined score = 0.40 × TF-IDF + 0.30 × BoW + 0.30 × Token Coverage.<br>Sources include real-time Wikipedia page extracts, local benchmark texts, and user-provided uploads.<br>Risk thresholds: HIGH ≥ 35%, MEDIUM ≥ 20%, LOW ≥ 10%, MINIMAL &lt; 10%.</div>', unsafe_allow_html=True)
