import streamlit as st
from supabase import create_client
import openai

st.set_page_config(page_title="Chat with Islamic Scholars", page_icon="☪️", layout="centered")

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
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
def get_channels():
    try:
        result = supabase.table('documents').select('metadata').limit(1000).execute()
        channels = list(set([r['metadata']['channel'] for r in result.data if 'channel' in r['metadata']]))
        return sorted(channels)
    except:
        return ["Muslim Orthodoxy", "Mohamed Hijab"]

channels = get_channels()
st.title("☪️ Chat with Islamic Scholars")
st.caption("Ask questions based on their YouTube content")

with st.sidebar:
    st.header("Settings")
    selected_channel = st.selectbox("Select Scholar:", channels, key="channel_select")
    
    st.markdown("---")
    st.markdown("### Advanced Settings")
    num_sources = st.slider("Number of sources to retrieve:", 3, 10, 7)
    response_length = st.slider("Max response length (tokens):", 500, 2000, 1200)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(f"Currently chatting with **{selected_channel}**")
    st.markdown("This chatbot uses RAG to answer based on actual video transcripts.")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

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
        with st.spinner("Thinking..."):
            try:
                # Get embedding for the question
                embedding_response = openai.embeddings.create(input=prompt, model="text-embedding-3-small")
                query_embedding = embedding_response.data[0].embedding
                
                # Retrieve more sources for better quality
                results = supabase.rpc('match_documents', {
                    'query_embedding': query_embedding, 
                    'match_count': num_sources
                }).execute()
                
                # Filter by selected channel
                filtered_results = [r for r in results.data if r['metadata'].get('channel') == selected_channel]
                
                # If no results for channel, use all results
                if not filtered_results:
                    filtered_results = results.data[:num_sources]
                
                if not filtered_results:
                    st.error("No relevant content found. Try a different question.")
                    st.stop()
                
                # Build context and sources
                context_parts = []
                sources = []
                
                for r in filtered_results:
                    context_parts.append(r['content'])
                    if 'title' in r['metadata'] and 'url' in r['metadata']:
                        sources.append({
                            'title': r['metadata']['title'],
                            'url': r['metadata']['url'],
                            'duration': r['metadata'].get('duration_str', 'Unknown')
                        })
                
                context = "\n\n---\n\n".join(context_parts)
                persona = filtered_results[0]['metadata'].get('persona', 'Islamic scholar')
                
                system_message = f"""You are {selected_channel}, {persona}.

Answer questions based ONLY on the following context from your YouTube videos:

{context}

Guidelines:
- Answer as if you are {selected_channel} speaking directly
- Use "I" and "my" when referring to your views and videos
- Stay true to the content and style shown in the videos
- If the answer isn't in the context, say "I haven't discussed that topic in detail in my videos yet."
- Be conversational, engaging, and thorough
- Provide detailed explanations when appropriate
- You can use the full context - don't hold back on detail"""

                # Use GPT-4o for best quality (128K context window)
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
                
                # Stream the response
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Show sources with more detail
                if sources:
                    with st.expander("📚 Sources Used"):
                        unique_sources = []
                        seen_urls = set()
                        for source in sources:
                            if source['url'] not in seen_urls:
                                unique_sources.append(source)
                                seen_urls.add(source['url'])
                        
                        for i, source in enumerate(unique_sources, 1):
                            st.markdown(f"**{i}. [{source['title']}]({source['url']})**")
                            st.caption(f"Duration: {source['duration']}")
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "context_length_exceeded" in str(e):
                    st.info("💡 Try reducing the 'Number of sources' slider in the sidebar.")
                else:
                    st.info("Please check your API keys and database connection.")