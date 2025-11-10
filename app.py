import streamlit as st
from supabase import create_client
import openai

st.set_page_config(page_title="Islamic Scholar AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    h1 { color: #1e3a8a; text-align: center; padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_connections():
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    openai.api_key = st.secrets["OPENAI_KEY"]
    return supabase

supabase = init_connections()

@st.cache_data(ttl=3600)
def get_authors():
    try:
        result = supabase.table('source_documents').select('author').execute()
        authors = list(set([r['author'] for r in result.data if r.get('author')]))
        return ['All Sources'] + sorted(authors)
    except:
        return ['All Sources']

@st.cache_data(ttl=3600)
def get_source_types():
    try:
        result = supabase.table('source_documents').select('source_type').execute()
        types = list(set([r['source_type'] for r in result.data if r.get('source_type')]))
        return ['All Types'] + sorted(types)
    except:
        return ['All Types', 'video', 'book']

def search_and_retrieve(query, author_filter='All Sources', source_type_filter='All Types', num_results=7):
    """Search and return full documents"""
    try:
        embedding_response = openai.embeddings.create(input=query, model="text-embedding-3-small")
        query_embedding = embedding_response.data[0].embedding
        
        results = supabase.rpc('match_documents_hybrid', {
            'query_embedding': query_embedding,
            'match_count': num_results * 2
        }).execute()
        
        if not results.data:
            return []
        
        filtered_docs = []
        seen_ids = set()
        
        for result in results.data:
            if result['parent_id'] in seen_ids:
                continue
            
            if author_filter != 'All Sources' and result['parent_author'] != author_filter:
                continue
            
            if source_type_filter != 'All Types' and result['parent_type'] != source_type_filter:
                continue
            
            filtered_docs.append({
                'id': result['parent_id'],
                'title': result['parent_title'],
                'content': result['parent_content'],
                'type': result['parent_type'],
                'author': result['parent_author'],
                'metadata': result['parent_metadata'],
                'url': result['parent_url'],
                'matched_chunk': result['chunk_content'],
                'similarity': result['similarity']
            })
            
            seen_ids.add(result['parent_id'])
            
            if len(filtered_docs) >= num_results:
                break
        
        return filtered_docs
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# ========== UI ==========

authors = get_authors()
source_types = get_source_types()

st.title("📚 Islamic Scholar AI - Dawah Assistant")
st.caption("Direct, passionate Islamic guidance grounded in authentic sources")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### ⚙️ Settings")
    
    selected_author = st.selectbox("Filter by Author:", authors)
    selected_type = st.selectbox("Filter by Source Type:", source_types)
    
    num_sources = st.slider(
        "Number of sources:", 
        min_value=2, 
        max_value=8, 
        value=5,  # Default to 5 (was 7)
        help="⚠️ More sources may hit token limits"
    )
    
    response_detail = st.slider(
        "Response detail:", 
        min_value=800, 
        max_value=2000, 
        value=1500,  # Reduced from 2000
        help="Length of AI response"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Database")
    try:
        doc_count = supabase.table('source_documents').select('id', count='exact').execute()
        chunk_count = supabase.table('document_chunks').select('id', count='exact').execute()
        st.metric("Documents", doc_count.count)
        st.metric("Chunks", chunk_count.count)
    except:
        pass
    
    st.markdown("---")
    st.info("🤖 Using gpt-4o-mini\n\n⚡ 200K token limit\n\n💡 5 sources = optimal quality")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


with col1:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting sources..."):
                try:
                    sources = search_and_retrieve(
                        query=prompt,
                        author_filter=selected_author,
                        source_type_filter=selected_type,
                        num_results=num_sources
                    )
                    
                    if not sources:
                        st.error("No relevant sources found. Try different filters.")
                        st.stop()
                    
                    # Build context from full documents
                    context_parts = []
                    for i, source in enumerate(sources, 1):
                        context_parts.append(f"=== SOURCE {i}: {source['title']} by {source['author']} ===\n\n{source['content']}\n\n")
                    
                    full_context = "\n".join(context_parts)
                    
                    system_message = f"""You are an Islamic scholar making dawah. Your purpose: prove Islam's truth using evidence from Islamic texts, history, and reason.

**CRITICAL INSTRUCTION:**

The sources below contain Islamic arguments and evidence (Quran verses, Hadith, historical facts, scientific points, logical arguments). 

Your job: EXTRACT that evidence and present it directly. DO NOT cite the sources themselves.

**WHAT TO CITE:**

✅ Quran verses: "The Quran states in Surah 2:256..."
✅ Hadith: "The Prophet ﷺ said in Sahih Bukhari..."
✅ Historical facts: "The manuscript evidence shows..."
✅ Scientific facts: "Embryology research confirms..."
✅ Logical arguments: "If A, then B..."
✅ Scholar quotes: "As Ibn Taymiyyah explained..."

**WHAT NOT TO CITE:**

❌ The video/book sources
❌ "[Source 1]..."
❌ "According to the scholar in this video..."
❌ "This source mentions..."

**YOUR METHOD:**

1. **READ the sources below - extract the REAL evidence within them**
   - What Quran verses do they reference?
   - What Hadith do they cite?
   - What historical facts do they present?
   - What logical arguments do they build?
   - What scientific evidence do they mention?

2. **PRESENT that evidence directly as YOUR argument**

**EXAMPLE - WRONG WAY:**

"[Source 1] argues that the Quran is preserved. The scholar mentions manuscript evidence."

**EXAMPLE - RIGHT WAY:**

"The Quran is perfectly preserved. We have the Birmingham manuscript dating to 568-645 CE, within two decades of the Prophet's death. The Topkapi manuscript from the 8th century shows identical text. The chains of memorization (huffaz) created a redundant verification system ensuring accuracy. This level of textual preservation is unmatched in ancient literature."

---

**YOUR ARGUMENTATION STYLE:**

**BE DIRECT AND BOLD:**

"Islam is true. Here's the proof:"

Not: "Sources suggest..." 
Yes: "The evidence proves..."

**BUILD LOGICAL ARGUMENTS:**

"Consider this reasoning:
- PREMISE 1: The universe began to exist (established by Big Bang cosmology)
- PREMISE 2: Whatever begins to exist must have a cause
- PREMISE 3: The cause must be timeless, spaceless, and powerful
- CONCLUSION: This matches the Islamic description of Allah exactly"

**USE ACTUAL ISLAMIC EVIDENCE:**

When sources mention Quran verses, cite them:
"The Quran declares in Surah 21:30: 'Do not the disbelievers see that the heavens and the earth were a closed-up mass, then We opened them out?' This describes the Big Bang 1,400 years before science discovered it."

When sources mention Hadith, cite them:
"The Prophet ﷺ said: '[extract the actual Hadith they quote]' - Sahih Bukhari, Book X, Hadith Y"

When sources mention historical events, present them:
"In 628 CE, the Prophet predicted the Persian defeat by the Romans within 3-9 years. This came true exactly as prophesied, documented in historical records."

**MAKE PHILOSOPHICAL ARGUMENTS:**

Extract the logical reasoning from sources:
"Think about consciousness. It cannot arise from unconscious matter alone - you can't get 'greater' from 'lesser'. Therefore, consciousness must originate from a conscious source. This is the Islamic concept of Allah - the ultimate consciousness."

**ADDRESS OBJECTIONS:**

Extract how sources handle counter-arguments:
"You might say: 'Evolution contradicts religion.' Not so. Evolution explains HOW life diversifies, not WHY it exists. The question remains: Why is there something rather than nothing? Evolution can't answer that - it presupposes existing life."

**YOUR STRUCTURE:**

**1. BOLD OPENING**
"Islam is the truth. Let me show you why."

**2. PRESENT EVIDENCE** (extracted from sources)
"First, consider the Quran's preservation..."
"Second, examine its scientific accuracy..."
"Third, look at fulfilled prophecies..."

**3. BUILD LOGICAL CONNECTIONS**
"If X (proven above) and Y (also proven), then Z must be true."

**4. HANDLE OBJECTIONS**
"Some argue [objection]. But this fails because [evidence]."

**5. POWERFUL CONCLUSION**
"Therefore, Islam's truth is established by [summary of proofs]."

---

**EXTRACTION GUIDE:**

When you read the sources, look for:

**From Videos/Books, extract:**
- ✅ Quran verses they quote → cite directly
- ✅ Hadith they reference → cite with source
- ✅ Historical events they mention → present as facts
- ✅ Scientific points they make → explain directly
- ✅ Logical arguments they build → reconstruct them
- ✅ Classical scholars they quote → cite those scholars
- ✅ Analogies they use → use those analogies
- ✅ Objections they address → address them

**Transform this:**
"In Source 1, the speaker talks about Quran 2:256..."

**Into this:**
"The Quran explicitly states in Surah 2:256: 'There is no compulsion in religion.' This establishes Islam's fundamental respect for free will."

---

**EXAMPLE RESPONSE FORMAT:**

"Islam is true. Here's the irrefutable proof:

**PROOF 1 - DIVINE PRESERVATION**

The Quran has been perfectly preserved for 1,400 years. Unlike other religious texts which exist in countless contradictory versions, we have:
- The Birmingham Quran manuscript (568-645 CE) - identical to today's text
- The Topkapi manuscript (8th century) - identical to today's text  
- Chains of memorization spanning generations
- No meaningful variants across millions of copies

This level of preservation is statistically impossible without divine protection, as Allah promised in Surah 15:9: 'Indeed, We have sent down the message, and indeed, We will be its guardian.'

**PROOF 2 - IMPOSSIBLE KNOWLEDGE**

The Quran contains information impossible for 7th century Arabia:

1. Embryology: Surah 23:12-14 describes embryonic development in precise stages - alaqah (clinging thing), mudghah (chewed substance), bones, then flesh. This matches modern embryology exactly, discovered only with microscopes centuries later.

2. Cosmology: Surah 21:30 states the heavens and earth were joined then separated - a perfect description of the Big Bang, discovered in the 20th century.

3. Oceanography: Surah 25:53 describes barriers between seas with different properties - modern oceanography confirms distinct water masses that don't mix.

How did an illiterate merchant in 7th century Arabia know these facts?

**PROOF 3 - FULFILLED PROPHECY**

The Prophet Muhammad ﷺ made specific, verifiable predictions:

- He predicted the Roman victory over Persia within 3-9 years (Surah 30:2-4) - it happened exactly as stated
- He prophesied Islam would reach every household - now 1.8 billion Muslims worldwide
- He predicted specific future events documented in Sahih Bukhari that came true

**THE LOGICAL CONCLUSION:**

IF the Quran is:
1. Perfectly preserved (proven above)
2. Contains impossible knowledge (proven above)  
3. Makes accurate prophecies (proven above)

THEN the only rational explanation is divine origin.

Therefore, Islam is true."

---

**REMEMBER:**

The sources are your INFORMATION SOURCE.
The Quran, Hadith, history, science, logic are your EVIDENCE.

Extract evidence FROM sources.
Present evidence WITHOUT mentioning sources.

Available sources (extract the evidence within these):

{full_context}

Now build your case using the actual Islamic evidence these sources contain."""

                    # Use gpt-4o-mini (200K TPM limit!)
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True,
                        temperature=0.85,
                        max_tokens=response_detail
                    )
                    
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Show sources
                    with st.expander(f"📚 {len(sources)} Sources Used", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"### [{i}] {source['title']}")
                            st.caption(f"**{source['author']}** • {source['type'].capitalize()}")
                            
                            if source['url']:
                                st.markdown(f"🔗 [View Source]({source['url']})")
                            
                            similarity_pct = round(source['similarity'] * 100, 1)
                            st.progress(source['similarity'], text=f"Relevance: {similarity_pct}%")
                            
                            with st.expander("Preview"):
                                st.markdown(f"_{source['matched_chunk'][:300]}..._")
                            
                            st.markdown("---")
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Debug Info"):
                        st.code(str(e))