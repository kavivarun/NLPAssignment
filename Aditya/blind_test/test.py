
# import google.generativeai as genai

# # Replace with your actual API key
# API_KEY = "AIzaSyDi3754bhB6bJn9Ifh62SyOKVzn2Tv2vB0"

# # Configure the API
# genai.configure(api_key=API_KEY)


# text = input("Input:")

# # Load the Gemini model
# model = genai.GenerativeModel("gemini-2.0-flash")

# # Prompt sent to Gemini
# prompt = f"""Respond with only the actual answer, no explanations or conext need in the response.Improve the following text by correcting grammatical errors, 
# removing filler words, and making it more clear and professional:

# "{text}"
# """


# response = model.generate_content(prompt)
# print(f"REULT:\n{response}")

# print(response._result.candidates[0].content.parts[0].text)


# ////////////////////////////

import streamlit as st
import pandas as pd
import random
from datetime import datetime
import io
import json

# Page config
st.set_page_config(page_title="Model Response Blind Test", page_icon="🔍", layout="wide")

# Initialize session state
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'randomized_data' not in st.session_state:
    st.session_state.randomized_data = []
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

def process_csv(df):
    """Process the uploaded CSV file and randomize responses"""
    # Find text column (case insensitive)
    text_cols = [col for col in df.columns if 'text' in col.lower()]
    if not text_cols:
        st.error("No column containing 'text' found in the CSV")
        return None
    
    text_col = text_cols[0]
    
    # Get model columns (assume all other columns are model responses)
    model_cols = [col for col in df.columns if col != text_col]
    
    if len(model_cols) < 4:
        st.error(f"Found only {len(model_cols)} model columns. Need at least 4.")
        return None
    
    # Take first 4 model columns
    model_cols = model_cols[:4]
    
    # Process each row
    processed_data = []
    for idx, row in df.iterrows():
        text = row[text_col]
        if pd.isna(text):
            continue
            
        # Create response list with original ordering
        responses = []
        for model in model_cols:
            response_text = row[model]
            if pd.isna(response_text):
                response_text = "[No response]"
            responses.append({
                'model': model,
                'response': response_text
            })
        
        # Create randomized version
        randomized = responses.copy()
        random.shuffle(randomized)
        
        # Create label mapping
        labels = ['A', 'B', 'C', 'D']
        label_mapping = {}
        for i, resp in enumerate(randomized):
            label_mapping[labels[i]] = resp['model']
        
        processed_data.append({
            'index': idx,
            'text': text,
            'original_responses': responses,
            'randomized_responses': randomized,
            'label_mapping': label_mapping
        })
    
    return processed_data

def analyze_group_results():
    """Analyze results from all users"""
    if not st.session_state.users:
        return None
    
    completed_users = {name: data for name, data in st.session_state.users.items() 
                      if data.get('completed', False)}
    
    if not completed_users:
        return None
    
    # Count selections per model across all users
    model_counts = {}
    question_analysis = []
    
    for i in range(len(st.session_state.randomized_data)):
        item = st.session_state.randomized_data[i]
        question_result = {
            'text': item['text'],
            'votes': {},
            'consensus': None
        }
        
        for user_name, user_data in completed_users.items():
            if i < len(user_data['selections']) and user_data['selections'][i]:
                selected_label = user_data['selections'][i]
                selected_model = item['label_mapping'][selected_label]
                
                # Count for overall statistics
                model_counts[selected_model] = model_counts.get(selected_model, 0) + 1
                
                # Count for this question
                question_result['votes'][selected_model] = question_result['votes'].get(selected_model, 0) + 1
        
        # Determine consensus (if any)
        if question_result['votes']:
            max_votes = max(question_result['votes'].values())
            consensus_models = [model for model, votes in question_result['votes'].items() if votes == max_votes]
            if len(consensus_models) == 1:
                question_result['consensus'] = consensus_models[0]
        
        question_analysis.append(question_result)
    
    return {
        'completed_users': list(completed_users.keys()),
        'total_users': len(completed_users),
        'overall_model_counts': model_counts,
        'question_analysis': question_analysis
    }

def main():
    st.title("Blind Test - Model Responses")
    
    # Upload CSV
    if st.session_state.csv_data is None:
        st.subheader("Upload your CSV file")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.csv_data = df
                st.session_state.randomized_data = process_csv(df)
                st.success("CSV loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    else:
        # User selection/creation
        st.sidebar.header("User Selection")
        
        # Show existing users
        existing_users = list(st.session_state.users.keys())
        
        if existing_users:
            st.sidebar.subheader("Existing Users")
            for user in existing_users:
                is_completed = st.session_state.users[user].get('completed', False)
                status = "✅ Completed" if is_completed else "⏳ In Progress"
                if st.sidebar.button(f"{user} ({status})"):
                    st.session_state.current_user = user
                    st.session_state.current_index = 0
                    st.rerun()
        
        # Add new user
        st.sidebar.subheader("Add New User")
        new_user_name = st.sidebar.text_input("Enter your name")
        if st.sidebar.button("Start Test") and new_user_name:
            st.session_state.current_user = new_user_name
            st.session_state.users[new_user_name] = {
                'selections': [None] * len(st.session_state.randomized_data),
                'completed': False,
                'started_at': datetime.now().isoformat()
            }
            st.session_state.current_index = 0
            st.rerun()
        
        # Show analysis button if there are completed users
        completed_count = sum(1 for user_data in st.session_state.users.values() 
                            if user_data.get('completed', False))
        if completed_count > 0:
            st.sidebar.subheader("View Results")
            if st.sidebar.button("Show Group Analysis"):
                st.session_state.current_user = "ANALYSIS"
                st.rerun()
        
        # Main content area
        if st.session_state.current_user == "ANALYSIS":
            # Show group analysis
            st.header("Group Analysis")
            analysis = analyze_group_results()
            
            if analysis:
                # Overall statistics
                st.subheader(f"Results from {analysis['total_users']} users")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Overall Model Preferences:**")
                    for model, count in sorted(analysis['overall_model_counts'].items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / sum(analysis['overall_model_counts'].values())) * 100
                        st.write(f"- {model}: {count} selections ({percentage:.1f}%)")
                
                with col2:
                    st.write("**Participants:**")
                    for user in analysis['completed_users']:
                        st.write(f"- {user} ✅")
                
                # Question-by-question analysis
                st.subheader("Case-by-Case Results")
                
                for i, qa in enumerate(analysis['question_analysis'], 1):
                    with st.expander(f"Original Text {i}: {qa['text'][:100]}..."):
                        if qa['votes']:
                            st.write("**Votes:**")
                            for model, votes in qa['votes'].items():
                                st.write(f"- {model}: {votes} vote(s)")
                            
                            if qa['consensus']:
                                st.success(f"**Consensus: {qa['consensus']}**")
                            else:
                                st.info("No consensus reached")
                        else:
                            st.info("No votes recorded for this question")
                
                # Export button
                if st.button("Export Group Results"):
                    results_data = []
                    for i, qa in enumerate(analysis['question_analysis'], 1):
                        row = {
                            'question_number': i,
                            'text': qa['text'],
                            'consensus': qa['consensus'] or 'No consensus'
                        }
                        
                        # Add vote counts for each model
                        for model in set().union(*[qa['votes'] for qa in analysis['question_analysis']]):
                            row[f"{model}_votes"] = qa['votes'].get(model, 0)
                        
                        # Add individual user selections
                        for user in analysis['completed_users']:
                            user_data = st.session_state.users[user]
                            if i-1 < len(user_data['selections']) and user_data['selections'][i-1]:
                                selected_label = user_data['selections'][i-1]
                                selected_model = st.session_state.randomized_data[i-1]['label_mapping'][selected_label]
                                row[f"{user}_selection"] = selected_model
                        
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"group_blind_test_results_{timestamp}.csv"
                    
                    st.download_button(
                        label="Download Group Results",
                        data=csv_buffer.getvalue(),
                        file_name=filename,
                        mime="text/csv"
                    )
            else:
                st.info("No completed tests yet. Results will appear when users finish their tests.")
        
        elif st.session_state.current_user:
            # Individual test view
            user_data = st.session_state.users[st.session_state.current_user]
            
            if user_data.get('completed', False):
                st.success(f"Test completed by {st.session_state.current_user}!")
                st.info("Go to Group Analysis to see combined results.")
                return
            
            st.header(f"Test for: {st.session_state.current_user}")
            
            # Navigation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Previous", disabled=st.session_state.current_index == 0):
                    st.session_state.current_index -= 1
                    st.rerun()
            
            with col2:
                progress = (st.session_state.current_index + 1) / len(st.session_state.randomized_data)
                st.progress(progress, text=f"Original Text {st.session_state.current_index + 1} of {len(st.session_state.randomized_data)}")
            
            with col3:
                if st.button("Next", disabled=st.session_state.current_index >= len(st.session_state.randomized_data) - 1):
                    st.session_state.current_index += 1
                    st.rerun()
            
            # Display current question
            current_item = st.session_state.randomized_data[st.session_state.current_index]
            
            st.write("**Original Text:**")
            st.write(current_item['text'])
            
            # Selection
            st.write("**Choose your preferred response:**")
            labels = ['A', 'B', 'C', 'D']
            current_selection = user_data['selections'][st.session_state.current_index]
            
            selected = st.radio("Your choice:", labels, 
                              index=labels.index(current_selection) if current_selection else None, 
                              horizontal=True)
            
            # Display responses
            st.write("**Responses:**")
            cols = st.columns(2)
            
            for i, resp in enumerate(current_item['randomized_responses']):
                with cols[i % 2]:
                    label = labels[i]
                    
                    # Simple styling based on selection
                    if selected == label:
                        st.info(f"**Response {label}** ✓")
                    else:
                        st.write(f"**Response {label}**")
                    
                    # Display response text with better formatting
                    st.write(resp['response'])
                    st.write("---")
            
            # Update selection if changed
            if selected != current_selection:
                user_data['selections'][st.session_state.current_index] = selected
                st.rerun()
            
            # Complete test button
            if st.session_state.current_index == len(st.session_state.randomized_data) - 1:
                completed_count = sum(1 for sel in user_data['selections'] if sel is not None)
                
                if completed_count == len(st.session_state.randomized_data):
                    if st.button("Complete Test", type="primary", use_container_width=True):
                        user_data['completed'] = True
                        user_data['completed_at'] = datetime.now().isoformat()
                        st.success("Test completed! Your results have been saved.")
                        st.info("Go to the sidebar and click 'Show Group Analysis' to see combined results.")
        else:
            st.info("Please select a user from the sidebar or add a new user to begin the test.")

if __name__ == "__main__":
    main()