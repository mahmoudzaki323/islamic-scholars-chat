import streamlit as st
from supabase import create_client
import openai

# Page config
st.set_page_config(
    page_title="Chat with Islamic Scholars",
    page_icon="☪️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize connections
@st.cache_resource
def init_connections():
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    openai.api_key = st.secrets["OPENAI_KEY"]
    return supabase

supabase = init_connections()

# Get available channels
@st.cache_data(ttl=3600)
def get_channels():
    try:
        result = supabase.table('documents').select('metadata').limit(1000).execute()
        channels = list(set([r['metadata']['channel'] for r in result.data if 'channel' in r['metadata']]))
        return sorted(channels)
    except:
        return ["Muslim Orthodoxy", "Mohamed Hijab"]

channels = get_channels()

# UI
st.title("☪️ Chat with Islamic Scholars")
st.caption("Ask questions based on their YouTube content")

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_channel = st.selectbox("Select Scholar:", channels, key="channel_select")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(f"Currently chatting with **{selected_channel}**")
    st.markdown("This chatbot uses RAG (Retrieval Augmented Generation) to answer based on actual video transcripts.")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get embedding for the question
                embedding_response = openai.embeddings.create(
                    input=prompt,
                    model="text-embedding-3-small"
                )
                query_embedding = embedding_response.data[0].embedding
                
                # Search Supabase for relevant content
                results = supabase.rpc('match_documents', {
                    'query_embedding': query_embedding,
                    'match_count': 5
                }).execute()
                
                # Filter by selected channel
                filtered_results = [
                    r for r in results.data 
                    if r['metadata'].get('channel') == selected_channel
                ]
                
                # If no results for selected channel, use top results from any channel
                if not filtered_results:
                    filtered_results = results.data[:3]
                
                if not filtered_results:
                    st.error("No relevant content found. Try a different question.")
                    st.stop()
                
                # Build context from results
                context_parts = []
                sources = []
                
                for r in filtered_results:
                    context_parts.append(r['content'])
                    if 'title' in r['metadata'] and 'url' in r['metadata']:
                        sources.append({
                            'title': r['metadata']['title'],
                            'url': r['metadata']['url']
                        })
                
                context = "\n\n---\n\n".join(context_parts)
                
                # Get persona
                persona = filtered_results[0]['metadata'].get('persona', 'Islamic scholar')
                
                # Generate response with streaming
                system_message = f"""You are {selected_channel}, {persona}.

Answer questions based ONLY on the following context from your YouTube videos:

{context}

Guidelines:
- Answer as if you are {selected_channel} speaking directly
- Use "I" and "my" when referring to your views and videos
- Stay true to the content and style shown in the videos
- If the answer isn't in the context, say "I haven't discussed that topic in detail in my videos yet."
- Be conversational and engaging
- Keep responses concise but informative"""

                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=800
                )
                
                # Stream the response
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Show sources
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources[:3], 1):
                            st.markdown(f"{i}. [{source['title']}]({source['url']})")
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check your API keys and database connection.")
```

**Commit this file.**

---

### **File 2: `requirements.txt`**

Click **"Add file"** → **"Create new file"** → Name it `requirements.txt`
```
streamlit
supabase
openai