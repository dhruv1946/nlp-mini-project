from io import BytesIO
import os
import json
import re
import urllib.request
import urllib.parse
from html.parser import HTMLParser
import socket
socket.setdefaulttimeout(3.5)
import math
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, brown
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim
_NLTK_PACKAGES = ['tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/stopwords', 'corpora/wordnet', 'corpora/omw-1.4', 'taggers/averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger_eng', 'taggers/universal_tagset', 'corpora/brown']
for _pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(_pkg)
    except LookupError:
        nltk.download(_pkg.split('/')[-1], quiet=True)
try:
    _nlp_spacy = spacy.load('en_core_web_sm')
except OSError:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')
    _nlp_spacy = spacy.load('en_core_web_sm')

class Preprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t.isalpha() and t.lower() not in self.stop_words]

    def lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(t.lower()) for t in tokens]

    def preprocess(self, text: str) -> dict:
        sentences = sent_tokenize(text)
        raw_tokens = word_tokenize(text)
        filtered_tokens = self.remove_stopwords(raw_tokens)
        lemmas = self.lemmatize(filtered_tokens)
        return {'sentences': sentences, 'raw_tokens': raw_tokens, 'filtered_tokens': filtered_tokens, 'lemmas': lemmas, 'num_sentences': len(sentences), 'num_tokens': len(raw_tokens), 'num_filtered': len(filtered_tokens), 'vocab_size': len(set(lemmas))}

class NGramAnalyzer:

    def __init__(self, tokens: list):
        self.tokens = [t.lower() for t in tokens if t.isalpha()]
        self.unigram_freq = Counter(self.tokens)
        self.vocab_size = len(self.unigram_freq)
        self.bigram_freq = Counter(ngrams(self.tokens, 2))
        self.trigram_freq = Counter(ngrams(self.tokens, 3))

    def laplace_bigram_prob(self, w1: str, w2: str) -> float:
        count_bigram = self.bigram_freq.get((w1.lower(), w2.lower()), 0)
        count_unigram = self.unigram_freq.get(w1.lower(), 0)
        return (count_bigram + 1) / (count_unigram + self.vocab_size)

    def sentence_log_prob(self, sentence_tokens: list) -> float:
        tokens = [t.lower() for t in sentence_tokens if t.isalpha()]
        if len(tokens) < 2:
            return 0.0
        return sum((math.log(self.laplace_bigram_prob(tokens[i - 1], tokens[i])) for i in range(1, len(tokens))))

    def get_all_stats(self) -> dict:
        return {'unigrams': self.unigram_freq.most_common(15), 'bigrams': self.bigram_freq.most_common(10), 'trigrams': self.trigram_freq.most_common(10), 'vocab_size': self.vocab_size, 'total_tokens': len(self.tokens)}

class HMMPosTagger:

    def __init__(self):
        self._train()

    def _train(self):
        try:
            tagged_sents = list(brown.tagged_sents(tagset='universal'))
        except LookupError:
            nltk.download('universal_tagset', quiet=True)
            nltk.download('brown', quiet=True)
            tagged_sents = list(brown.tagged_sents(tagset='universal'))
        tag_counts = defaultdict(int)
        trans_counts = defaultdict(lambda: defaultdict(int))
        emiss_counts = defaultdict(lambda: defaultdict(int))
        initial_counts = defaultdict(int)
        all_words = set()
        for sent in tagged_sents:
            if not sent:
                continue
            initial_counts[sent[0][1]] += 1
            for i, (word, tag) in enumerate(sent):
                w_lower = word.lower()
                all_words.add(w_lower)
                tag_counts[tag] += 1
                emiss_counts[tag][w_lower] += 1
                if i > 0:
                    trans_counts[sent[i - 1][1]][tag] += 1
        self.vocab_size = len(all_words)
        self.tags = list(tag_counts.keys())
        total_sents = sum(initial_counts.values())
        num_tags = len(self.tags)
        self.initial_log_probs = {tag: math.log((initial_counts.get(tag, 0) + 1) / (total_sents + num_tags)) for tag in self.tags}
        self.trans_log_probs = {}
        for prev_tag in self.tags:
            total = sum(trans_counts[prev_tag].values())
            self.trans_log_probs[prev_tag] = {tag: math.log((trans_counts[prev_tag].get(tag, 0) + 1) / (total + num_tags)) for tag in self.tags}
        self.emiss_log_probs = {}
        for tag in self.tags:
            total = sum(emiss_counts[tag].values())
            oov_prob = math.log(1 / (total + self.vocab_size))
            tag_emiss = defaultdict(lambda p=oov_prob: p)
            for word, count in emiss_counts[tag].items():
                tag_emiss[word] = math.log((count + 1) / (total + self.vocab_size))
            self.emiss_log_probs[tag] = tag_emiss

    def viterbi(self, tokens: list) -> list:
        words = [t.lower() for t in tokens]
        n = len(words)
        if n == 0:
            return []
        viterbi_table = [{}]
        backpointer = [{}]
        for tag in self.tags:
            emit = self.emiss_log_probs[tag][words[0]]
            viterbi_table[0][tag] = self.initial_log_probs.get(tag, -1e+18) + emit
            backpointer[0][tag] = None
        for t in range(1, n):
            vt = {}
            bp = {}
            for tag in self.tags:
                emit = self.emiss_log_probs[tag][words[t]]
                best_prob = -1e+18
                best_prev = self.tags[0]
                for prev in self.tags:
                    prob = viterbi_table[t - 1][prev] + self.trans_log_probs[prev][tag] + emit
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev
                vt[tag] = best_prob
                bp[tag] = best_prev
            viterbi_table.append(vt)
            backpointer.append(bp)
        best_final = max(self.tags, key=lambda t: viterbi_table[n - 1][t])
        tag_seq = [best_final]
        for t in range(n - 1, 0, -1):
            tag_seq.append(backpointer[t][tag_seq[-1]])
        tag_seq.reverse()
        return list(zip(tokens, tag_seq))

    def tag(self, text: str) -> list:
        tokens = [t for t in word_tokenize(text) if t.isalpha() or t in '.!?,;:']
        return self.viterbi(tokens)

class CYKParser:
    GRAMMAR = {'S': [('NP', 'VP'), ('NP', 'ADJP')], 'NP': [('DT', 'NN'), ('DT', 'NNS'), ('DT', 'NNP'), ('DT', 'NNPS'), ('JJ', 'NN'), ('JJ', 'NNS'), ('CD', 'NNS'), ('DT', 'ADJP'), ('NP', 'PP'), ('NP', 'NNS'), ('PRP',), ('NNP',), ('NN',), ('NNS',)], 'VP': [('VB', 'NP'), ('VBZ', 'NP'), ('VBD', 'NP'), ('VBP', 'NP'), ('VBN', 'NP'), ('VBG', 'NP'), ('MD', 'VP'), ('VP', 'PP'), ('VP', 'NP'), ('VP', 'ADJP'), ('VB',), ('VBZ',), ('VBD',), ('VBP',), ('VBN',), ('VBG',)], 'PP': [('IN', 'NP'), ('TO', 'NP')], 'ADJP': [('RB', 'JJ'), ('JJ',), ('VBN',)], 'ADVP': [('RB',)]}
    TERMINAL_MAP = defaultdict(set)
    BINARY_RULES = []
    for _lhs, _rhs_list in GRAMMAR.items():
        for _rhs in _rhs_list:
            if len(_rhs) == 1:
                TERMINAL_MAP[_rhs[0]].add(_lhs)
            elif len(_rhs) == 2:
                BINARY_RULES.append((_lhs, _rhs[0], _rhs[1]))

    def parse(self, pos_tags: list) -> dict:
        n = len(pos_tags)
        if n == 0:
            return {'is_valid': False}
        table = [[set() for _ in range(n)] for _ in range(n)]
        for i, tag in enumerate(pos_tags):
            table[i][i] = set(self.TERMINAL_MAP.get(tag, set()))
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    for lhs, B, C in self.BINARY_RULES:
                        if B in table[i][k] and C in table[k + 1][j]:
                            table[i][j].add(lhs)
        return {'is_valid': 'S' in table[0][n - 1]}

class ChunkExtractor:
    NP_GRAMMAR = '\n        NP: {<DT|PRP\\$>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}\n            {<PRP>}\n    '
    VP_GRAMMAR = '\n        VP: {<MD>?<VB|VBD|VBG|VBN|VBP|VBZ><RP>?}\n    '

    def __init__(self):
        self.np_parser = nltk.RegexpParser(self.NP_GRAMMAR)
        self.vp_parser = nltk.RegexpParser(self.VP_GRAMMAR)

    def extract(self, pos_tagged: list) -> dict:
        np_tree = self.np_parser.parse(pos_tagged)
        vp_tree = self.vp_parser.parse(pos_tagged)
        noun_phrases = [' '.join((w for w, _ in subtree.leaves())) for subtree in np_tree.subtrees(filter=lambda t: t.label() == 'NP')]
        verb_phrases = [' '.join((w for w, _ in subtree.leaves())) for subtree in vp_tree.subtrees(filter=lambda t: t.label() == 'VP')]
        return {'noun_phrases': noun_phrases, 'verb_phrases': verb_phrases}

class NERExtractor:

    def __init__(self):
        self.nlp = _nlp_spacy

    def extract(self, text: str) -> dict:
        doc = self.nlp(text)
        entities = [{'text': ent.text, 'label': ent.label_, 'description': spacy.explain(ent.label_) or ent.label_, 'start': ent.start_char, 'end': ent.end_char} for ent in doc.ents]
        html = displacy.render(doc, style='ent', page=False, options={'colors': {'PERSON': '#4A90D9', 'ORG': '#7ED321', 'GPE': '#F5A623', 'MONEY': '#BD10E0', 'DATE': '#9B9B9B', 'TIME': '#50E3C2', 'LOC': '#E8D44D', 'PRODUCT': '#E94560', 'EVENT': '#FF6B6B'}})
        return {'entities': entities, 'html': html, 'entity_count': len(entities)}

class DocumentParser:

    @staticmethod
    def extract_text(file_data, filename: str) -> str:
        if hasattr(file_data, 'getvalue'):
            raw_bytes = file_data.getvalue()
        elif hasattr(file_data, 'read'):
            raw_bytes = file_data.read()
        elif isinstance(file_data, (bytes, bytearray)):
            raw_bytes = bytes(file_data)
        else:
            raw_bytes = b''
        ext = filename.lower().split('.')[-1]
        if ext == 'pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(BytesIO(raw_bytes))
                return '\n'.join((page.extract_text() for page in reader.pages if page.extract_text())).strip()
            except Exception:
                return ''
        elif ext in ['docx', 'doc']:
            try:
                import docx
                doc = docx.Document(BytesIO(raw_bytes))
                return '\n'.join((p.text for p in doc.paragraphs if p.text)).strip()
            except Exception:
                return ''
        else:
            try:
                return raw_bytes.decode('utf-8').strip()
            except UnicodeDecodeError:
                return raw_bytes.decode('latin-1', errors='ignore').strip()

class _DDGHtmlParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.in_snippet = False
        self.snippets = []
        self.cur_text = []
        self.in_title = False
        self.titles = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'td' and attrs_dict.get('class') == 'result-snippet':
            self.in_snippet = True
            self.cur_text = []
        elif tag == 'a' and attrs_dict.get('class') == 'result-link':
            self.in_title = True
            self.cur_text = []

    def handle_endtag(self, tag):
        if tag == 'td' and self.in_snippet:
            self.in_snippet = False
            self.snippets.append(''.join(self.cur_text).strip())
        elif tag == 'a' and self.in_title:
            self.in_title = False
            self.titles.append(''.join(self.cur_text).strip())

    def handle_data(self, data):
        if self.in_snippet or self.in_title:
            self.cur_text.append(data)

class PlagiarismDetector:
    _TFIDF_WEIGHT = 0.4
    _BOW_WEIGHT = 0.3
    _COVERAGE_WEIGHT = 0.3

    @staticmethod
    def extract_search_queries(text: str, entities: list=None, max_queries: int=8, *args, **kwargs) -> list:
        raw_sents = [s.strip() for s in re.split(r'[\n.?!;]+', text) if len(s.strip().split()) >= 4]
        queries = []
        if not raw_sents:
            words = text.strip().split()
            if len(words) >= 4:
                queries.append(' '.join(words[:10]))
        else:
            step = max(1, len(raw_sents) // min(len(raw_sents), 5))
            for s in raw_sents[::step][:6]:
                words = [w for w in s.split() if w]
                if len(words) <= 10:
                    queries.append(' '.join(words))
                else:
                    queries.append(' '.join(words[:9]))
                    if len(words) >= 16:
                        mid = len(words) // 2
                        queries.append(' '.join(words[mid:mid + 8]))
        if entities:
            for ent in entities:
                if ent.get('label') in ('ORG', 'PERSON', 'GPE', 'WORK_OF_ART', 'PRODUCT', 'EVENT') and len(ent.get('text', '')) > 2:
                    queries.append(ent['text'])
        dedup = []
        seen = set()
        for q in queries:
            clean = re.sub(r'[^\w\s]', '', q).strip()
            if len(clean.split()) >= 2 and clean.lower() not in seen:
                seen.add(clean.lower())
                dedup.append(clean)
            if len(dedup) >= max_queries:
                break
        return dedup

    def analyze(self, submitted_text: str, reference_texts: list, ref_names: list, source_types: list=None) -> dict:
        if not reference_texts:
            return {'results': [], 'sources_count': 0, 'error': 'No reference documents provided.'}
        corpus = [submitted_text] + reference_texts
        tfidf_matrix = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(corpus)
        bow_matrix = CountVectorizer(stop_words='english', min_df=1).fit_transform(corpus)
        tfidf_sims = sk_cosine_sim(tfidf_matrix[0], tfidf_matrix[1:])[0]
        bow_sims = sk_cosine_sim(bow_matrix[0], bow_matrix[1:])[0]
        sub_words = set((w.lower() for w in word_tokenize(submitted_text) if w.isalpha()))
        sub_len = max(1, len(sub_words))
        results = []
        for idx, (ref_text, ref_name) in enumerate(zip(reference_texts, ref_names)):
            tfidf_sim = float(tfidf_sims[idx])
            bow_sim = float(bow_sims[idx])
            ref_words = set((w.lower() for w in word_tokenize(ref_text) if w.isalpha()))
            coverage = len(sub_words.intersection(ref_words)) / sub_len
            combined = round((self._TFIDF_WEIGHT * tfidf_sim + self._BOW_WEIGHT * bow_sim + self._COVERAGE_WEIGHT * coverage) * 100, 2)
            stype = source_types[idx] if source_types and idx < len(source_types) else 'Reference File'
            results.append({'reference': ref_name, 'source_type': stype, 'tfidf_score': round(tfidf_sim * 100, 2), 'bow_score': round(bow_sim * 100, 2), 'coverage_score': round(coverage * 100, 2), 'combined': combined, 'risk_level': self._classify_risk(combined)})
        results.sort(key=lambda r: r['combined'], reverse=True)
        return {'results': results, 'sources_count': len(results), 'error': None}

    def fetch_web_sources(self, query_terms: list, max_results: int=8) -> list:
        sources = []
        seen_titles = set()
        for term in query_terms:
            if len(sources) >= max_results:
                break
            clean = term.strip(' "\'.,;:()[]{}')
            if not clean or len(clean) < 10:
                continue
            try:
                url = 'https://lite.duckduckgo.com/lite/'
                data = urllib.parse.urlencode({'q': clean}).encode('utf-8')
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    html_content = r.read().decode('utf-8', errors='ignore')
                parser = _DDGHtmlParser()
                parser.feed(html_content)
                for title, snippet in zip(parser.titles, parser.snippets):
                    t_lower = title.lower()
                    if t_lower in seen_titles:
                        continue
                    seen_titles.add(t_lower)
                    if len(snippet) > 40:
                        sources.append({'reference': f'Web // {title[:35]}', 'text': snippet, 'source_type': 'Live Web'})
                    if len(sources) >= max_results:
                        break
            except Exception:
                continue
        return sources

    def fetch_academic_sources(self, query_terms: list, max_results: int=4) -> list:
        headers = {'User-Agent': 'AcademicPlagiarismDetector/1.0 (mailto:academic.nlp@education.org)'}
        sources = []
        seen_titles = set()
        for term in query_terms:
            if len(sources) >= max_results:
                break
            clean = term.strip(' "\'.,;:()[]{}')
            if not clean or len(clean) < 8:
                continue
            try:
                url = f'https://api.openalex.org/works?search={urllib.parse.quote(clean)}&per-page=2'
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3.5) as r:
                    data = json.loads(r.read().decode('utf-8'))
                for res in data.get('results', []):
                    title = res.get('title', '')
                    if not title or title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    host = res.get('primary_location', {}).get('source', {}) or {}
                    venue = host.get('display_name', 'Scholarly Publisher')
                    inv_idx = res.get('abstract_inverted_index') or {}
                    abstract = ' '.join(w for _, w in sorted(((pos, w) for w, pos_list in inv_idx.items() for pos in pos_list)))
                    content = (title + '. ' + abstract).strip()
                    if len(content) > 60:
                        sources.append({'reference': f'Academic // {title[:35]} ({venue[:20]})', 'text': content, 'source_type': 'Academic Paper'})
                    if len(sources) >= max_results:
                        break
            except Exception:
                pass
            if len(sources) < max_results:
                try:
                    c_url = f'https://api.crossref.org/works?query={urllib.parse.quote(clean)}&rows=2'
                    c_req = urllib.request.Request(c_url, headers=headers)
                    with urllib.request.urlopen(c_req, timeout=3.5) as cr:
                        c_data = json.loads(cr.read().decode('utf-8'))
                    for it in c_data.get('message', {}).get('items', []):
                        title = it.get('title', [''])[0]
                        if not title or title.lower() in seen_titles:
                            continue
                        seen_titles.add(title.lower())
                        pub = it.get('publisher', 'Academic Publisher')
                        raw_ab = it.get('abstract', '')
                        clean_ab = re.sub(r'<[^>]+>', '', raw_ab).strip()
                        content = (title + '. ' + clean_ab).strip()
                        if len(content) > 60:
                            sources.append({'reference': f'Crossref // {title[:35]} ({pub[:20]})', 'text': content, 'source_type': 'Academic Paper'})
                        if len(sources) >= max_results:
                            break
                except Exception:
                    pass
        return sources

    def fetch_wikipedia_sources(self, query_terms: list, max_pages: int=6) -> list:
        headers = {'User-Agent': 'ContentQualityEvaluator/1.0 (NLP Educational Project; student@example.edu)'}
        sources = []
        seen_titles = set()
        for term in query_terms:
            if len(sources) >= max_pages:
                break
            clean_term = term.strip(' "\'.,;:()[]{}')
            if not clean_term or len(clean_term) < 3:
                continue
            try:
                url = f'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_term)}&srlimit=2&format=json'
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    data = json.loads(r.read().decode('utf-8'))
                hits = data.get('query', {}).get('search', [])
                for hit in hits:
                    title = hit.get('title', '')
                    if not title or title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    p_url = f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exchars=4000&titles={urllib.parse.quote(title)}&format=json'
                    p_req = urllib.request.Request(p_url, headers=headers)
                    with urllib.request.urlopen(p_req, timeout=3.0) as pr:
                        p_data = json.loads(pr.read().decode('utf-8'))
                    pages = p_data.get('query', {}).get('pages', {})
                    page = next(iter(pages.values()), {})
                    extract = page.get('extract', '')
                    if extract and len(extract) > 60:
                        sources.append({'reference': f'Wikipedia // {title}', 'text': extract, 'source_type': 'Live Wikipedia'})
                    if len(sources) >= max_pages:
                        break
            except Exception:
                continue
        return sources

    def load_local_corpus(self, corpus_dir: str='corpus') -> list:
        sources = []
        if not os.path.exists(corpus_dir):
            return sources
        valid_exts = ('.txt', '.pdf', '.docx', '.doc')
        for fname in os.listdir(corpus_dir):
            if fname.lower().endswith(valid_exts):
                fpath = os.path.join(corpus_dir, fname)
                try:
                    with open(fpath, 'rb') as f:
                        raw_bytes = f.read()
                    content = DocumentParser.extract_text(raw_bytes, fname)
                    if content:
                        clean_name = os.path.splitext(fname)[0].replace('_', ' ').title()
                        sources.append({'reference': f'Corpus // {clean_name}', 'text': content, 'source_type': 'Benchmark Corpus'})
                except Exception:
                    continue
        return sources

    def analyze_hybrid(self, submitted_text: str, uploaded_files: list=None, search_terms: list=None, corpus_dir: str='corpus') -> dict:
        references = []
        if uploaded_files:
            for f in uploaded_files:
                try:
                    text = DocumentParser.extract_text(f, f.name)
                    if text:
                        references.append({'reference': f'Uploaded // {f.name}', 'text': text, 'source_type': 'Uploaded File'})
                except Exception:
                    continue
        if not search_terms and submitted_text.strip():
            search_terms = self.extract_search_queries(submitted_text, max_queries=8)
        if search_terms:
            references.extend(self.fetch_web_sources(search_terms[:4], max_results=6))
            references.extend(self.fetch_academic_sources(search_terms[:3], max_results=4))
            references.extend(self.fetch_wikipedia_sources(search_terms[:3], max_pages=3))
        references.extend(self.load_local_corpus(corpus_dir))
        valid_refs = [r for r in references if r.get('text')]
        if not valid_refs:
            return {'results': [], 'sources_count': 0, 'error': 'No reference sources available.'}
        return self.analyze(submitted_text, [r['text'] for r in valid_refs], [r['reference'] for r in valid_refs], [r['source_type'] for r in valid_refs])

    @staticmethod
    def _classify_risk(score: float) -> str:
        if score >= 35:
            return 'HIGH'
        elif score >= 20:
            return 'MEDIUM'
        elif score >= 10:
            return 'LOW'
        else:
            return 'MINIMAL'
