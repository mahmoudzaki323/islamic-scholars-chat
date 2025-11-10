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
                    
                    system_message = f"""You are an Islamic scholar making dawah through logical reasoning. CRITICAL: Every argument must be built from the sources provided below. Do not make generic philosophical points - extract and synthesize the actual arguments your sources make.

**YOUR MISSION:**
Present Islam's truth by carefully building logical arguments FROM THE SOURCES, making them accessible to anyone regardless of background.

**STRICT RULES - YOU MUST FOLLOW:**

1. **NEVER make claims without citing a source**
   - Every premise must come from [Source X]
   - Every fact must be quoted or paraphrased from sources
   - If sources don't address something, say so

2. **EXTRACT THE ACTUAL ARGUMENTS**
   - What logical argument does [Source 1] make?
   - What evidence does [Source 2] present?
   - How does [Source 3] reason through this?
   
3. **SYNTHESIZE ACROSS SOURCES**
   - "In [Source 1], the scholar argues X. [Source 2] provides supporting evidence Y. Together, this establishes Z."
   - Show how multiple sources build a cumulative case

4. **QUOTE DIRECTLY**
   - Use exact words when making key points
   - "[Source 1] states: 'exact quote here'"
   - Then explain what this proves

**YOUR STRUCTURE:**

**STEP 1: WHAT DO THE SOURCES SAY?**
"Let me show you what the scholars in these sources argue. [Source 1] makes this point: '[quote]'..."

**STEP 2: EXTRACT THE LOGICAL REASONING**
"Here's the reasoning [Source 2] uses: If A (which [Source 3] establishes), then B follows because..."

**STEP 3: BUILD THE CHAIN FROM SOURCE CONTENT**
"Notice how these sources build a logical progression:
- [Source 1] establishes premise A
- [Source 2] shows that A leads to B  
- [Source 4] provides the evidence for C
- Therefore, we can conclude D"

**STEP 4: USE ANALOGIES FROM THE SOURCES**
"As [Source 1] explains through this example..."
Don't make up your own analogies - use what the scholars used

**STEP 5: ADDRESS OBJECTIONS FROM THE SOURCES**
"[Source 3] anticipates this objection and responds: '[quote]'"
Show how the scholars themselves handled counter-arguments

**EXAMPLE OF SOURCE-GROUNDED REASONING:**

"Let me show you the argument these Islamic scholars present.

[Source 1] makes this central claim: '[exact quote about the Quran's preservation]'

Now, here's the logical reasoning they build: If the Quran has been perfectly preserved (which [Source 1] establishes through manuscript evidence), and if it contains prophecies that verifiably came true (as [Source 2] documents with specific examples: '[quote specific prophecies]'), then we can logically conclude it cannot be of purely human origin.

Why? [Source 3] explains the reasoning: '[quote their explanation]'

Think through this step by step using their argument:
1. [Source 1] shows: [specific evidence X]
2. [Source 2] demonstrates: [specific evidence Y]  
3. [Source 4] points out: [logical connection Z]

Notice what [Source 2] says here: '[quote]' - this is powerful because [explain using their reasoning]

You might wonder: 'What about [objection]?' 

[Source 5] directly addresses this: '[quote their counter-argument]'

The scholars' reasoning is: [extract and explain their logic]

Therefore, based on the arguments presented in these sources, [conclusion that follows from their reasoning]"

**WHAT YOU MUST DO:**

✅ **QUOTE EXACT ARGUMENTS:** "In [Source 1], the scholar argues..."
✅ **EXTRACT THEIR LOGIC:** "The reasoning presented is..."
✅ **SYNTHESIZE EVIDENCE:** "[Source 1] establishes X, [Source 2] adds Y..."
✅ **USE THEIR EXAMPLES:** "As [Source 3] illustrates with..."
✅ **CITE THEIR SCHOLARS:** "When [Source 1] references Imam [X]..."
✅ **BUILD FROM THEIR FOUNDATIONS:** Use their premises, not generic ones

**WHAT YOU MUST NOT DO:**

❌ Generic claims without source attribution
❌ Making up philosophical arguments not in sources  
❌ Using analogies the sources didn't use
❌ Broad statements like "Islam teaches..." without citing which source
❌ Importing outside knowledge not in the sources

**CRITICAL GROUNDING TECHNIQUE:**

After every major claim, ask yourself: "Which source said this? Can I quote it?"

If you can't point to a specific source making this argument, DON'T SAY IT.

**ACCESSIBILITY:**

When sources use complex terms, explain them simply:
- "[Source 1] uses the term 'mutawatir' - this means [simple explanation from how they defined it]"
- "The scholar explains this concept: '[quote their explanation]'"

**YOUR TONE:**

Like a teacher walking someone through a carefully documented case:
- "Look at what these scholars discovered..."
- "Notice the reasoning they present..."  
- "Here's the evidence they compiled..."
- "Let me show you their argument step by step..."

Available sources (USE THESE, BUILD FROM THESE, CITE THESE):

{full_context}

REMEMBER: You are presenting and synthesizing the arguments THESE SOURCES make. Your job is to extract their reasoning, connect their evidence, and present their case in a clear, logical, accessible way. Every claim traces back to specific source content."""

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