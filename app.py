import streamlit as st
from supabase import create_client
import openai

st.set_page_config(page_title="Islamic Scholar AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    h1 { color: #1e3a8a; text-align: center; padding: 1rem 0; }
    .source-card { 
        background: #f8fafc; 
        padding: 1rem; 
        border-radius: 8px; 
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .evidence-block {
        background: #fef3c7;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #f59e0b;
    }
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
    """Get list of available scholars/authors"""
    try:
        result = supabase.table('source_documents').select('author').execute()
        authors = list(set([r['author'] for r in result.data if r.get('author')]))
        return ['All Sources'] + sorted(authors)
    except:
        return ['All Sources']

@st.cache_data(ttl=3600)
def get_source_types():
    """Get list of source types"""
    try:
        result = supabase.table('source_documents').select('source_type').execute()
        types = list(set([r['source_type'] for r in result.data if r.get('source_type')]))
        return ['All Types'] + sorted(types)
    except:
        return ['All Types', 'video', 'book']

def search_and_retrieve(query, author_filter='All Sources', source_type_filter='All Types', num_results=5):
    """
    Hybrid retrieval:
    1. Search chunks to find relevant documents
    2. Return FULL parent documents (not just chunks)
    """
    try:
        # Create embedding for query
        embedding_response = openai.embeddings.create(input=query, model="text-embedding-3-small")
        query_embedding = embedding_response.data[0].embedding
        
        # Search chunks and get parent documents
        results = supabase.rpc('match_documents_hybrid', {
            'query_embedding': query_embedding,
            'match_count': num_results * 2  # Get more to filter
        }).execute()
        
        if not results.data:
            return []
        
        # Apply filters
        filtered_docs = []
        seen_ids = set()
        
        for result in results.data:
            # Skip duplicates
            if result['parent_id'] in seen_ids:
                continue
            
            # Apply author filter
            if author_filter != 'All Sources' and result['parent_author'] != author_filter:
                continue
            
            # Apply source type filter
            if source_type_filter != 'All Types' and result['parent_type'] != source_type_filter:
                continue
            
            filtered_docs.append({
                'id': result['parent_id'],
                'title': result['parent_title'],
                'content': result['parent_content'],  # FULL document content
                'type': result['parent_type'],
                'author': result['parent_author'],
                'metadata': result['parent_metadata'],
                'url': result['parent_url'],
                'matched_chunk': result['chunk_content'],  # Just for showing what matched
                'similarity': result['similarity']
            })
            
            seen_ids.add(result['parent_id'])
            
            if len(filtered_docs) >= num_results:
                break
        
        return filtered_docs
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def format_evidence(sources):
    """Format sources for display"""
    evidence = []
    for i, source in enumerate(sources, 1):
        source_type = source['type'].capitalize()
        author = source['author']
        title = source['title']
        
        if source['url']:
            evidence.append(f"**[{i}] {author} - {title}** ({source_type})")
            evidence.append(f"🔗 {source['url']}")
        else:
            evidence.append(f"**[{i}] {author} - {title}** ({source_type})")
        
        # Show similarity score
        similarity_pct = round(source['similarity'] * 100, 1)
        evidence.append(f"📊 Relevance: {similarity_pct}%")
        evidence.append("")
    
    return "\n".join(evidence)

# ========== UI ==========

authors = get_authors()
source_types = get_source_types()

st.title("📚 Islamic Scholar AI")
st.caption("Engage in scholarly discourse grounded in source texts")

# Two-column layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### ⚙️ Settings")
    
    selected_author = st.selectbox("Filter by Author:", authors)
    selected_type = st.selectbox("Filter by Source Type:", source_types)
    num_sources = st.slider("Number of sources:", 3, 10, 5)
    response_length = st.slider("Response detail:", 800, 2500, 1500)
    
    st.markdown("---")
    st.markdown("### 📊 Database")
    try:
        doc_count = supabase.table('source_documents').select('id', count='exact').execute()
        chunk_count = supabase.table('document_chunks').select('id', count='exact').execute()
        st.metric("Full Documents", doc_count.count)
        st.metric("Searchable Chunks", chunk_count.count)
    except:
        pass
    
    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

with col1:
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching sources and formulating response..."):
                try:
                    # Step 1: Search and retrieve FULL documents
                    sources = search_and_retrieve(
                        query=prompt,
                        author_filter=selected_author,
                        source_type_filter=selected_type,
                        num_results=num_sources
                    )
                    
                    if not sources:
                        st.error("No relevant sources found. Try adjusting filters or ask a different question.")
                        st.stop()
                    
                    # Step 2: Build context from FULL documents
                    context_parts = []
                    for i, source in enumerate(sources, 1):
                        # Include FULL content, not just chunks
                        context_parts.append(f"=== SOURCE {i}: {source['title']} by {source['author']} ===\n{source['content']}\n")
                    
                    full_context = "\n".join(context_parts)
                    
                    # Step 3: Scholar system prompt
                    system_message = f"""You are a knowledgeable Islamic scholar engaged in rigorous intellectual discourse.

Your methodology:
1. **Make clear arguments** - State your position directly
2. **Cite specific evidence** - Reference exact sources with [Source X]
3. **Show reasoning** - Explain the logical chain: Evidence → Inference → Conclusion
4. **Be precise** - Quote or paraphrase specific passages when making points
5. **Acknowledge complexity** - Present multiple views when they exist
6. **Engage directly** - Respond to the question, don't just summarize

Available sources (FULL documents):

{full_context}

Critical instructions:
- You are NOT roleplaying as any specific person
- You are a scholar analyzing and synthesizing these sources
- Make arguments like: "The evidence suggests... because Source 2 states... Therefore..."
- Use format: [Source 1], [Source 2], etc. when citing
- If sources disagree, acknowledge both perspectives
- If sources don't address the question, say so clearly

Respond in a scholarly but accessible tone. Be direct and argumentative when appropriate."""

                    # Step 4: Generate response with streaming
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True,
                        temperature=0.7,
                        max_tokens=response_length
                    )
                    
                    # Stream response
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Step 5: Show sources used
                    with st.expander("📚 Sources Consulted", expanded=True):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"### [{i}] {source['title']}")
                            st.caption(f"**Author:** {source['author']} | **Type:** {source['type'].capitalize()}")
                            
                            if source['url']:
                                st.markdown(f"🔗 [View Source]({source['url']})")
                            
                            similarity_pct = round(source['similarity'] * 100, 1)
                            st.progress(source['similarity'], text=f"Relevance: {similarity_pct}%")
                            
                            # Show what part matched (the chunk that triggered this document)
                            with st.expander("Preview matched section"):
                                st.markdown(f"_{source['matched_chunk'][:300]}..._")
                            
                            st.markdown("---")
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if "context_length_exceeded" in str(e):
                        st.info("💡 Try reducing the 'Number of sources' - some documents are very long.")
                    else:
                        st.info("Check console for details.")
                    import traceback
                    st.code(traceback.format_exc())