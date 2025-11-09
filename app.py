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

def smart_truncate(text, max_words=8000):
    """Keep more content for better quality"""
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Take 80% from beginning, 20% from end
    first_part = int(max_words * 0.8)
    last_part = int(max_words * 0.2)
    
    truncated = ' '.join(words[:first_part]) + "\n\n[... content truncated ...]\n\n" + ' '.join(words[-last_part:])
    return truncated

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
            
            # Smart truncate but keep more content
            truncated_content = smart_truncate(result['parent_content'], max_words=8000)
            
            filtered_docs.append({
                'id': result['parent_id'],
                'title': result['parent_title'],
                'content': truncated_content,
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
    num_sources = st.slider("Number of sources:", 3, 10, 7)
    response_detail = st.slider("Response detail:", 1000, 3000, 2000)
    
    st.markdown("---")
    st.markdown("### 📊 Database")
    try:
        doc_count = supabase.table('source_documents').select('id', count='exact').execute()
        chunk_count = supabase.table('document_chunks').select('id', count='exact').execute()
        st.metric("Documents", doc_count.count)
        st.metric("Searchable Chunks", chunk_count.count)
    except:
        pass
    
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
            with st.spinner("Consulting sources and preparing response..."):
                try:
                    sources = search_and_retrieve(
                        query=prompt,
                        author_filter=selected_author,
                        source_type_filter=selected_type,
                        num_results=num_sources
                    )
                    
                    if not sources:
                        st.error("No relevant sources found. Try different filters or ask another question.")
                        st.stop()
                    
                    # Build rich context
                    context_parts = []
                    for i, source in enumerate(sources, 1):
                        context_parts.append(f"=== SOURCE {i}: {source['title']} by {source['author']} ({source['type']}) ===\n\n{source['content']}\n\n")
                    
                    full_context = "\n".join(context_parts)
                    
                    # Passionate dawah system prompt
                    system_message = f"""You are a knowledgeable and passionate Islamic scholar whose purpose is dawah - inviting people to understand, appreciate, and embrace Islam.

Your style and methodology:

**BE DIRECT AND CONFIDENT**
- Don't hedge: "Islam teaches..." not "Some Muslims believe..."
- State the truth clearly and with conviction
- Make definitive arguments backed by evidence

**BE WARM AND INVITING**
- Speak like a caring teacher who genuinely wants the best for the questioner
- Show the beauty, wisdom, and mercy of Islam
- Help them fall in love with the deen through your words

**BE EVIDENCE-BASED**
- Always cite sources: [Source 1], [Source 2], etc.
- Quote directly when making key points
- Show the proof before the conclusion
- Build arguments: Evidence → Reasoning → Conclusion

**BE SCHOLARLY BUT ACCESSIBLE**
- Use proper Islamic terminology (explain when needed)
- Reference scholars, hadith, Quran precisely
- But speak in a way anyone can understand and appreciate

**YOUR MISSION**
This is not academic debate - this is dawah. Your goal is to:
1. Clarify misconceptions about Islam
2. Present irrefutable evidence for Islam's truth
3. Inspire love for Allah and His religion
4. Remove doubts and strengthen faith
5. Invite to the straight path with wisdom and beautiful preaching

**STRUCTURE YOUR RESPONSES LIKE A SHEIKH**
1. Direct answer to their question (1-2 sentences)
2. Primary evidence from sources (Quran, Hadith, scholars)
3. Logical reasoning showing why this is true
4. Additional supporting evidence
5. Powerful conclusion that ties it together

Available sources:

{full_context}

Remember: You are not giving a Wikipedia summary. You are a passionate scholar making dawah. Be confident, warm, evidence-based, and inspiring. Make them see the truth and beauty of Islam through your words.

May Allah guide us all to the straight path."""

                    # Use GPT-4o for maximum quality
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True,
                        temperature=0.85,  # Higher for more passionate responses
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
                    with st.expander("📚 Sources Consulted", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"### [{i}] {source['title']}")
                            st.caption(f"**{source['author']}** • {source['type'].capitalize()}")
                            
                            if source['url']:
                                st.markdown(f"🔗 [View Source]({source['url']})")
                            
                            similarity_pct = round(source['similarity'] * 100, 1)
                            st.progress(source['similarity'], text=f"Relevance: {similarity_pct}%")
                            
                            with st.expander("Preview matched section"):
                                st.markdown(f"_{source['matched_chunk'][:400]}..._")
                            
                            st.markdown("---")
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
                    if "rate_limit" in str(e).lower():
                        st.warning("""
                        **Rate Limit Exceeded**
                        
                        Your OpenAI account has a 30K tokens/minute limit. This happened because:
                        - You're sending ~7 full documents (each can be 5K-10K tokens)
                        - Total request was too large
                        
                        **Quick fixes:**
                        1. Reduce "Number of sources" slider to 3-4
                        2. Wait 1 minute and try again
                        3. Upgrade your OpenAI tier at platform.openai.com
                        """)
                    
                    elif "context_length" in str(e).lower():
                        st.warning("Context too large. Reduce number of sources.")
                    
                    else:
                        with st.expander("Debug Info"):
                            st.code(str(e))