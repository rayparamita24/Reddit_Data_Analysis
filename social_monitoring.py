import streamlit as st
import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy import stats
import numpy as np
import re
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from collections import Counter
from nltk.tokenize import word_tokenize
# from page4 import run_page4
#from datetime import datetime,timedelta
import string 
import pandas as pd

from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
nltk.download('punkt_tab')
nltk.download('words')
# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#from langdetect import detect, DetectorFactory
#from langdetect.lang_detect_exception import LangDetectException
#from sklearn.cluster import DBSCAN
#from scipy import stats
#import pandas as pd
#from sklearn.feature_selection import VarianceThreshold
#from sklearn.impute import SimpleImputer
#from io import BytesIO
#from matplotlib.backends.backend_pdf import PdfPages
import textstat
from nltk.corpus import words as nltk_words  # Ensure to download the NLTK word corpus
#from collections import Counter
import json
from scipy.optimize import curve_fit
from rapidfuzz import fuzz



# Function to check for required columns
def check_required_columns(df):
    """
    Checks if at least one column from required groups exists in the DataFrame.
    If required columns are missing, displays an error and stops execution.

    Parameters:
        df (DataFrame): Input DataFrame.

    Returns:
        None
    """
    # Define groups of required columns
    required_groups = [
        ['comment body', 'comments'],  # At least one of these
        ['comment date']               # This column is mandatory
    ]

    missing_groups = []

    # Check if at least one column exists in each group
    for group in required_groups:
        if not any(col in df.columns for col in group):
            missing_groups.append(group)

    # If any group is missing, show error and stop execution
    if missing_groups:
        missing_columns_message = " or ".join([" | ".join(group) for group in missing_groups])
        st.error(f"The dataset must include the following columns: {missing_columns_message}")
        st.stop()

# Function to calculate deletion rate over time
def calculate_deletion_rate(df, text_column, date_column):
    """
    Calculates the deletion rate over time.

    Parameters:
        df (DataFrame): Input DataFrame.
        text_column (str): Name of the text column.
        date_column (str): Name of the date column.

    Returns:
        DataFrame: Aggregated deletion counts over time.
    """
    # Filter rows where comments are 'deleted' or 'removed'
    deleted_df = df[df[text_column].str.lower().isin(['[deleted]', '[removed]'])]
    
    # Convert the date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    deleted_df[date_column] = pd.to_datetime(deleted_df[date_column], errors='coerce')
    
    # Drop rows with invalid dates
    deleted_df = deleted_df.dropna(subset=[date_column])

    # Count deletions by date
    deletion_counts = deleted_df[date_column].dt.date.value_counts().sort_index()
    deletion_df = pd.DataFrame({'date': deletion_counts.index, 'deletion_count': deletion_counts.values})
    
    return deletion_df


    

def is_similar(post_title, keywords, threshold=70):
    # Check similarity of post_title with each keyword
    for keyword in keywords:
        similarity = fuzz.ratio(post_title.lower(), keyword.lower())
        if similarity >= threshold:  # If similarity is greater than or equal to the threshold, consider it relevant
            return True
    return False

# Function to clean and filter posts based on keywords and similarity
def clean_reddit_data(df, keywords, similarity_threshold=70):
    # 1. Remove posts with missing important data
    required_columns = ['post title', 'post url', 'post date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")

    # Step 2: Remove rows with missing important data
    df = df.dropna(subset=required_columns)
    
    # Step 3: Remove exact duplicates based on Post Title and Post URL
    df = df.drop_duplicates(subset=['post title', 'post url'], keep='first')
       
    # 3. Filter posts that contain relevant keywords in the Post Title or are similar
    df_relevant = df[df['post title'].apply(lambda x: is_similar(x, keywords, similarity_threshold))]
    
    return df_relevant

def find_exact_duplicates(df):
    duplicates = df[df.duplicated(subset=['post title', 'post url'], keep=False)]
    return duplicates

# Function to find near duplicates based on similarity
def find_near_duplicates(df, threshold=95):
    near_duplicates = []
    titles = df['post title'].tolist()

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            similarity = fuzz.ratio(titles[i], titles[j])
            if similarity >= threshold:
                near_duplicates.append((i, j, similarity))

    return near_duplicates

def count_exact_duplicates(df):
    duplicates = df.duplicated(subset=['post title', 'post url'], keep=False)
    total_exact_duplicates = duplicates.sum() // 2  # Divide by 2 to avoid counting pairs twice
    return total_exact_duplicates

# Function to count near-duplicate posts
def count_near_duplicates(df, threshold=80):
    titles = df['Post Title'].tolist()
    near_duplicates_set = set()

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            similarity = fuzz.ratio(titles[i], titles[j])
            if similarity >= threshold:
                near_duplicates_set.add((i, j))

    total_near_duplicates = len(near_duplicates_set)
    return total_near_duplicates



# Function to assess user credibility

def contains_links(post_content):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return bool(re.search(url_pattern, post_content))

# Function to simulate malware checking (by detecting suspicious URLs)
def is_too_short(post_content, min_length=3):
    return len(post_content.split()) <= min_length
def check_brand(post_content):
    # For actual malware detection, use an API like VirusTotal
    # This is a placeholder for detecting suspicious URLs in the post content
    suspicious_keywords = ['buy now', 'free money', 'click here', 'limited offer', 'get rich']
    
    return any(keyword in post_content.lower() for keyword in suspicious_keywords)

# Function to flag posts as spam
def flag_spam(df):
    spam_flags = []
    removed_keywords = ['[removed]', '[deleted]', '[censored]']
    
    for index, row in df.iterrows():
        # Try to fetch the first available content column
        post_content = row.get('Comment Body') or row.get('Post Title')or row.get('Comments') or row.get('comment body') or row.get('post title')or row.get('comments')
        
        # If no content found, skip this row
        if not post_content:
            spam_flags.append('No Content')
            continue
        
        spam_flag = 'Not Spam'  # Default flag
        
        # Check for embedded links
        if contains_links(post_content):
           spam_flag = 'Contains Embedded Link'
               
        # Check if the post is a removed or deleted message
        elif any(keyword in post_content.lower() for keyword in removed_keywords):
             spam_flag = 'Removed or Deleted Post'
        elif is_too_short(post_content):
             spam_flag = 'Too Short Message'
        elif check_brand(post_content):
             spam_flag = 'Contains spam words'
        # Check for repeated posts
        
    
        spam_flags.append(spam_flag)
    
    df['Spam Flag'] = spam_flags
    return df
# Visualization function to show the number of spam posts

# Exponential decay function
def exponential_decay(t, E0, lambd):
    return E0 * np.exp(-lambd * t)

# Function to calculate post decay rate for upvotes, downvotes, and post score
def calculate_decay_rate(df):
    decay_rates = {}

    # Define required column groups
    required_columns_group1 = ['upvotes', 'downvotes', 'Time Since Post']
    required_columns_group2 = ['post score', 'Time Since Post']

    # Check if at least one group of columns exists in the dataset
    if not (all(col in df.columns for col in required_columns_group1) or all(col in df.columns for col in required_columns_group2)):
        st.error("Neither of the required column groups (Upvotes/Downvotes or Post Score) are present.")
        return

    # Determine which group of columns is present
    if all(col in df.columns for col in required_columns_group1):
        columns_to_use = ['upvotes', 'downvotes']
    elif all(col in df.columns for col in required_columns_group2):
        columns_to_use = ['post score']
    else:
        st.error("Incomplete column group found. Please make sure you have either Upvotes & Downvotes or Post Score.")
        return

    # Ensure 'Time Since Post' is numeric
    if 'Time Since Post' in df.columns:
        df['Time Since Post'] = pd.to_numeric(df['Time Since Post'], errors='coerce')

    # Fit the exponential decay model to the available columns
    for metric in columns_to_use:
        if df['Time Since Post'].isnull().any() or df[metric].isnull().any():
            st.warning(f"Missing values detected in 'Time Since Post' or '{metric}', skipping this metric.")
            continue

        try:
            # Fit exponential decay model
            popt, _ = curve_fit(exponential_decay, df['Time Since Post'], df[metric], p0=(df[metric].iloc[0], 0.1))
            
            # Extract the parameters E0 (initial value) and lambda (decay rate)
            E0, lambd = popt
            decay_rates[metric] = lambd
            
            # Predict the decay of the metric over time
            df[f'Predicted {metric}'] = exponential_decay(df['Time Since Post'], *popt)
            
            # Plot the actual and predicted values over time
            plt.figure(figsize=(10, 6))
            plt.plot(df['Time Since Post'], df[metric], label=f'Actual {metric}', marker='o')
            plt.plot(df['Time Since Post'], df[f'Predicted {metric}'], label=f'Predicted Decay of {metric}', linestyle='--')
            plt.xlabel("Time Since Post (hours)")
            plt.ylabel(f"{metric}")
            plt.title(f"Post Engagement Decay ({metric})")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error fitting decay model for {metric}: {e}")


def is_alphanumeric(value):
    return isinstance(value, str) and value.isalnum()

# Function to check datetime format
from datetime import datetime

# Function to validate datetime values against multiple formats
def is_valid_datetime(value, formats):
    for fmt in formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue  # Try the next format
    return False

# Function to validate the dataset based on metadata
def check_consistency(df, metadata):
    issues = []
    
    required_columns = metadata["columns"].keys()
    
    # Define acceptable datetime formats
    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",  # ISO format with seconds
        "%d-%m-%Y %H:%M",     # Day-Month-Year with hour and minute
        "%d/%m/%Y",           # Day/Month/Year
        "%Y/%m/%d %H:%M",     # Year/Month/Day with hour and minute
        "%A, %d %B %Y"        # Full weekday, day, full month, year
    ]
    
    # Check if each required column is in the dataset
    for column in required_columns:
        if column not in df.columns:
            issues.append(f"Column '{column}' is missing from the dataset.")
            continue  # Skip this column if it's missing
        
        column_info = metadata["columns"].get(column, {})
        actual_type = str(df[column].dtype)

        # Check column data types and constraints
        if "type" in column_info:
            expected_type = column_info["type"]
            
            # Check integer type
            if expected_type == "int" and actual_type not in ["int64", "float64"]:
                issues.append(f"Column '{column}' should be of type {expected_type}, but found {actual_type}.")
            
            # Check string type
            elif expected_type == "str" and actual_type != "object":
                issues.append(f"Column '{column}' should be of type {expected_type}, but found {actual_type}.")
            
            # Check datetime type
            elif expected_type == "datetime":
                invalid_values = df[column].dropna().astype(str).apply(lambda x: not is_valid_datetime(x, datetime_formats))
                if invalid_values.any():
                    issues.append(f"Column '{column}' contains invalid datetime values.")
    
    return issues



def total_removed_deleted(df, values_to_check=['[removed]', '[deleted]']):
    # Check for occurrences in the entire DataFrame
    total_count = df.isin(values_to_check).sum().sum()
    return total_count


DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",    # Example: 2024-12-06 13:45:00
    "%d-%m-%Y %H:%M",        # Example: 06-12-2024 13:45
    "%d/%m/%Y",              # Example: 06/12/2024
    "%Y/%m/%d %H:%M",        # Example: 2024/12/06 13:45
    "%A, %d %B %Y",          # Example: Monday, 06 December 2024
]

# Function to parse datetime using multiple formats
def parse_datetime(value):
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue  # If the format doesn't work, try the next one
    return None  # Return None if no format matches

# Function to extract start and end post date
def extract_start_end_post_dates(df, post_date_column):
    # Check if the column exists
    if post_date_column not in df.columns:
        st.error(f"Column '{post_date_column}' not found in the dataset.")
        return None, None

    # Parse all dates in the "Post Date" column
    df['Parsed Post Date'] = df[post_date_column].apply(parse_datetime)

    # Filter out any rows with invalid dates (None)
    valid_dates = df['Parsed Post Date'].dropna()

    # Extract start and end post date (earliest and latest date)
    if not valid_dates.empty:
        start_post_date = valid_dates.min()  # Earliest date
        end_post_date = valid_dates.max()    # Latest date
        return start_post_date, end_post_date
    else:
        st.error("No valid Post Dates found.")
        return None, None



# File uploader



#slang_terms = {'Lit', 'Fam', 'YOLO', 'GOAT', ' Dm', 'Lit', 'OMG', 'Bussin', 'AMA', ' BTW', 'Drip', 'FOMO', 'Fire', ' IYKYK', 'IMHO', ' GTG or G2G'}
common_words = set(nltk_words.words())  # List of known English words

def check_interpretation_errors(df, unknown_word_threshold=50):
 #   total_slang_count = 0
    total_unknown_count = 0
  #  total_abbreviation_count = 0
    total_word_count = 0

    for column in df.columns:
        if df[column].dtype == 'object':  # Check if column is text-based
            text = ' '.join(df[column].dropna().astype(str))  # Combine all text in the column
            
            # Count words and calculate metrics
            words = re.findall(r'\b\w+\b', text.lower())
            word_count = len(words)
            total_word_count += word_count

            # Check for Slang Words
   #         slang_count = sum(1 for word in words if word in slang_terms)
    #        total_slang_count += slang_count

            # Check for Unknown Words
            unknown_count = sum(1 for word in words if word not in common_words)
            total_unknown_count += unknown_count

            # Check for Abbreviations
     #       abbreviation_count = sum(1 for word in words if word.isupper() and len(word) > 1)
      #      total_abbreviation_count += abbreviation_count

    # Calculate percentages
    #slang_percentage = (total_slang_count / total_word_count) * 100 if total_word_count else 0
    unknown_percentage = (total_unknown_count / total_word_count) * 100 if total_word_count else 0
    #abbreviation_percentage = (total_abbreviation_count / total_word_count) * 100 if total_word_count else 0

    # Determine statuses
    #slang_status = 'High Slang Usage' if slang_percentage > slang_threshold else 'Acceptable Slang Usage'
    unknown_words_status = 'High Ambiguous and Meaningless words' if unknown_percentage > unknown_word_threshold else 'Acceptable mount of Ambiguous and Meaningless words'
    #abbreviation_status = 'High Abbreviation Usage' if abbreviation_percentage > abbreviation_threshold else 'Acceptable Abbreviation Usage'

    # Store results in a summary
    interpretation_errors_summary = {
        #'Total Slang Count': total_slang_count,
        'Total Unknown Count': total_unknown_count,
        #'Total Abbreviation Count': total_abbreviation_count,
        #'Slang Status': slang_status,
        'Unknown Words Status': unknown_words_status,
       # 'Abbreviation Status': abbreviation_status,
        'Total Word Count': total_word_count
    }
    
    return interpretation_errors_summary
def calculate_total_score(df):
    total_scores = []  # To store the average readability scores for each post
    
    # Iterate through columns to check if they're text-based
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if column is text-based
            text = ' '.join(df[column].dropna().astype(str))  # Combine all text in the column
            try:
                # Calculate multiple readability scores for the text in the column
                flesch_score = textstat.flesch_reading_ease(text)
                fk_grade_level = textstat.flesch_kincaid_grade(text)
                gunning_fog = textstat.gunning_fog(text)
                smog_index = textstat.smog_index(text)
                ari_score = textstat.automated_readability_index(text)
                coleman_liau = textstat.coleman_liau_index(text)
                
                # Average of all the readability scores
                avg_score = (flesch_score + fk_grade_level + gunning_fog +
                             smog_index + ari_score + coleman_liau) / 6

                # Append the average score for the column (post)
                total_scores.append(avg_score)
            except Exception as e:
                print(f"Error processing column {column}: {str(e)}")
                total_scores.append(None)
    
    # Calculate the overall average readability score for the entire dataset
    overall_avg_score = sum(total_scores) / len(total_scores) if total_scores else None
    return overall_avg_score, total_scores







# Function to clean the text

def clean_text(text):
    if pd.isnull(text):
        return []
  
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def analyze_text(text):
    # Ensure the input is a string before analyzing
    if not isinstance(text, str):
        return "Neutral"
    
    # Extract sentiment using TextBlob
    blob = TextBlob(text)
    opinion = blob.sentiment.polarity  # Sentiment polarity

    # Determine sentiment
    if opinion > 0:
        sentiment = "Positive"
    elif opinion < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment

def summarize_sentiments_and_frequencies(df, text_column, n):
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0,
        "N-gram Frequencies": Counter(),
    }
    
    for text in df[text_column]:
        cleaned_tokens = clean_text(text)
        sentiment = analyze_text(text)

        # Update sentiment counts
        summary[sentiment] += 1
        
        # Generate N-grams
        n_grams = ngrams(cleaned_tokens, n)
        summary["N-gram Frequencies"].update(n_grams)
    
    # Convert N-gram frequencies from tuples to strings
    summary["N-gram Frequencies"] = { ' '.join(ngram): count for ngram, count in summary["N-gram Frequencies"].items() }

    return summary


def check_post_credibility(row):
    """
    Checks if a post is credible based on conditions:
    Post Score > 5, Number of Comments > 3, and NSFW is False.
    """
    # Safely fetch values from the row with a default fallback (None)
    post_score = row.get('Post Score') or row.get('post score')or row.get('Score') or row.get('score')
    num_comments = row.get('Number of Comments') or row.get('number of comments')
    is_nsfw = row.get('NSFW') or row.get('nsfw')
    upvote = row.get('upvote')
    downvote = row.get('downvote')

    if post_score is None and upvote is not None and downvote is not None:
        post_score = upvote - downvote  # Calculate post score

    # Validate that the required values are not None
    if post_score is not None and num_comments is not None and is_nsfw is not None:
        # Ensure all values are of correct types
        if isinstance(post_score, (int, float)) and isinstance(num_comments, (int, float)) and isinstance(is_nsfw, bool):
            # Check credibility conditions
            if post_score > 5 and num_comments > 3 and not is_nsfw:
                return True  # Post is credible
            else:
                return False  # Post is not credible
        else:
            return False  # Invalid types found in data
    else:
        return False  # Missing required values

def display():
    # st.title("Survey Data Monitoring")

    # File upload option
    uploaded_file = st.file_uploader(
        "Upload a social data file (CSV )", type=["csv"])

    if uploaded_file is not None:
        # Display the name of the uploaded file
       try:
           df = pd.read_csv(uploaded_file, encoding=encoding)
       except Exception:
    # Fallback to latin-1 if the detected encoding fails
           df = pd.read_csv(uploaded_file, encoding='latin-1')
           all_columns = df.columns.tolist()
       
           with st.expander("", expanded=True):
                st.subheader("Data Quality Metrics Overview")
                col1,col2,col3,col4= st.columns(4)
                        
           with col1:
               st.markdown("<span style='color:blue; font-weight:bold;'>Total Rows & Columns:</span>", unsafe_allow_html=True) 
               # dupliid_check(df)
               #dqi_metrics = calculate_dqi(df)
               #st.metric("Duplicate Rows", dqi_metrics['Duplicate Rows'])
               st.metric("Total Rows", len(df))
               st.metric("Total Columns", len(df.columns))

               
               st.markdown("<span style='color:blue; font-weight:bold;'>Freshness of Post</span>", unsafe_allow_html=True) 
                    
               start_date, end_date = extract_start_end_post_dates(df, 'Post Date')

               if start_date and end_date:
                   st.success(f"**Start Post Date**: {start_date}")
                   st.success(f"**End Post Date**: {end_date}")


                # Display "Interpretation Errors" heading in bold blue text
               st.markdown('<span style="color:blue; font-weight:bold; font-size:18px;">Interpretation Errors</span>', 
               unsafe_allow_html=True)

# Get summary from your function
               interpretation_errors_summary = check_interpretation_errors(df)
               #st.markdown("<span style='color:green; font-size:16px;'> Ambiguous and Meaningless representation:</span>", unsafe_allow_html=True) 
               st.markdown(f"<p style='color:black; font-weight:bold; font-size:14px;'>"
                           f"{interpretation_errors_summary['Unknown Words Status']}</p>", unsafe_allow_html=True)
               st.markdown(f"<p style='font-size:14px;'><b>Total Word Count:</b> "
                           f"{interpretation_errors_summary['Total Word Count']}</p>", unsafe_allow_html=True)               
               
               #st.markdown(f"<p style='font-size:14px;'><b>:</b> ", unsafe_allow_html=True)
               missing_values = df.isnull().sum()              
               missing_percent = (missing_values / len(df)) * 100
               missing_percent = df.isnull().mean().mean() * 100  # Average of missing values across columns

# Display the metric
               st.metric("Incomplete representation", f"{missing_percent:.2f}%")
               st.markdown("<span style='color:blue; font-weight:bold;'>Post Redundancy:</span>", unsafe_allow_html=True)
               df.columns = [col.lower() for col in df.columns]
               required_columns = ['post title', 'post url']
               required_columns1 = ['title']
               if not (all(col in df.columns for col in required_columns) or all(col in df.columns for col in required_columns1)):
                   st.error(f"CSV must include the following columns: {', '.join(required_columns)}")
               
    
               else:
                   # Exact duplicates
                  total_exact_duplicates = count_exact_duplicates(df)
                  #total_near_duplicates = count_near_duplicates(df, threshold=80)

                  st.write(f"Exact Redundant Post: {total_exact_duplicates}")
                  exact_duplicates = find_exact_duplicates(df)
                  near_duplicates = find_near_duplicates(df, threshold=95)

# Calculate the counts for the bar chart
                  exact_duplicate_count = len(exact_duplicates)
                  near_duplicate_count = len(near_duplicates)
                  fig, ax = plt.subplots(figsize=(8, 5))
                  ax.bar(['Exact Duplicates', 'Near Duplicates'], [exact_duplicate_count, near_duplicate_count], color=['blue', 'orange'])
                  ax.set_ylabel('Count')
                  ax.set_title('Duplicate and Near-Duplicate Post Counts')

               # Display the bar chart in Streamlit
                  st.pyplot(fig)


# Create a bar chart for duplicate counts


# Display Total Word Count
  
              

               with col2:
                   
                  st.markdown("<span style='color:blue; font-weight:bold;'>Total Post Deleted:</span>", unsafe_allow_html=True) 
                  df.columns = [col.lower() for col in df.columns]
                  required_columns = ['post title', 'post url']
                  required_columns1 = ['title']
                  if not (all(col in df.columns for col in required_columns) or all(col in df.columns for col in required_columns1)):
                      st.error(f"CSV must include the following columns: {', '.join(required_columns)}")
                  else:
                      total_count = total_removed_deleted(df)
                      st.markdown(f"<p style='color: green; font-size: 15px; font-weight: bold;'>Total Count: {total_count}</p>", unsafe_allow_html=True)

                  #st.metric("Unique Value Count",dqi_metrics['Unique Values'].sum())
                  st.markdown('<span style="color:blue; font-weight:bold;">Noisy Post Count</span>', unsafe_allow_html=True)
                  #df['Post Date'] = pd.to_datetime(df['Post Date'])
                  # Flag spam posts
                  required_columns = ['post title', 'comment body']
                  required_columns1 = ['title', 'comments']
                  if not (all(col in df.columns for col in required_columns) or all(col in df.columns for col in required_columns1)):
                      st.error(f"CSV must include the following columns: {', '.join(required_columns)}")
                  else:
                      df = flag_spam(df)
                  #st.write(df)
                  #visualize_spam(df)
                      total_spam = df[df['Spam Flag'] != 'Not Spam'].shape[0]
                  #total_spam = len(df[df['Spam Flag'] != 'Not Spam'])
                      st.write(f"Total Spam Count: {total_spam}")
                      spam_counts = df['Spam Flag'].value_counts()
                      st.write(spam_counts)
                  st.markdown('<span style="color:blue; font-weight:bold;">Post Credibility</span>', unsafe_allow_html=True)
                  df.columns = [col.lower() for col in df.columns]

                  required_columns = ['post title', 'post score','number of comments', 'nsfw']
                  required_columns1 = ['title', 'score']
                  if not (all(col in df.columns for col in required_columns) or all(col in df.columns for col in required_columns1)):
                      st.error(f"CSV must include the following columns: {', '.join(required_columns)}")

                  
                  else:
                      
                      dw = df.apply(check_post_credibility, axis=1)                     
                      credible_posts_count = dw.sum()
                      total_posts = len(df)
                      st.write(f"Total Posts: {total_posts}")

                      st.write(f"Number of credible posts: {credible_posts_count}")
  
                      
                  st.markdown("<span style='color:blue; font-weight:bold;'>Post Coverage Error:</span>", unsafe_allow_html=True)
                  keywords_input = st.text_area("Enter Proper data collection keyword")
                  required_columns = ['post title']
                  required_columns1 = ['title']
                  
                  keywords = [keyword.strip() for keyword in keywords_input.split(',')] if keywords_input else []
                   # Button to fetch and filter posts
                  if st.button('Check Post Coverage'):
                       if keywords:
                           st.write(f"Filtering posts for keywords: {', '.join(keywords)}")
                           if not (all(col in df.columns for col in required_columns) or all(col in df.columns for col in required_columns1)):
                               st.error(f"CSV must include the following columns: {', '.join(required_columns)}")
                           
                           else:

                               df_clean = clean_reddit_data(df, keywords)
                               if len(df_clean) >8:
                
                                  st.write(f"Coverage error is in acceptable range.")
                               else:
                                  st.write("Data Coverage not upto the mark or you have entered wrong keywords")
                       else:
                           st.warning("Please enter at least one keyword to filter posts.")
                           
                  
                      
                                           

# Create a bar chart for duplicate counts


# Count deleted posts by date
                     

                     # Streamlit interface
                     
                     # Plot deleted posts over time
                     
                   
                 
               with col3:
                  st.markdown("<span style='color:blue; font-weight:bold;'>Post Deletion Rate:</span>", unsafe_allow_html=True)
                  check_required_columns(df)
                  text_column = 'comment body' if 'comment body' in df.columns else 'comments'
                  date_column = 'comment date'
                  
                  #st.success(f"Using '{text_column}' as the text column and '{date_column}' as the date column.")

                  # Calculate deletion rate
                  #st.write("### Deletion Rate Calculation:")
                  deletion_df = calculate_deletion_rate(df, text_column, date_column)

                  if deletion_df.empty:
                      st.warning("No 'deleted' or 'removed' comments found in the dataset.")
                  else:
                     # st.write("Deletion Rate Data:")
                      st.write(deletion_df)
                      st.markdown("<span style='color:black; font-weight:bold; size:10px'>Deletion Rate Over Time</span>", unsafe_allow_html=True)
                      # Plot deletion rate over time
                      
                      plt.figure(figsize=(10, 6))
                      plt.plot(deletion_df['date'], deletion_df['deletion_count'], marker='o', linestyle='-', color='red')
                      plt.xlabel('Date')
                      plt.ylabel('Number of Deletions')
                      plt.title('Comment Deletion Rate Over Time')
                      plt.xticks(rotation=45)
                      plt.grid(True)
                      st.pyplot(plt)
                 

        # Determine available columns
                  st.markdown("<span style='color:blue; font-weight:bold;'>Post Readability Check:</span>", unsafe_allow_html=True) 
                  overall_score, post_scores = calculate_total_score(df)
                  if overall_score >= 40:
                     quality_message = "Post is Readable"
                     message_color = "green"
                  else:
                     quality_message = "Post not easily readable"
                     message_color = "red"

# Display the score with quality message and custom styling
                  st.markdown(f"""
    
    <p style="color: {message_color}; font-size: 15px; font-weight: bold;">
        {quality_message}
    </p>
""", unsafe_allow_html=True)
   # Display the overall readability score
                  st.markdown("<span style='color:blue; font-weight:bold;'>Consistency Check with reference Metadata:</span>", unsafe_allow_html=True)
                  option = st.radio("Choose Metadata Source:", ["Custom Metadata", "Upload JSON Metadata File"])
                  if option == "Custom Metadata":
                      custom_metadata = {
                          "columns": {
                              "post id": {"type": "int", "constraints": ["unique", "non-null"]},
                              "comment id": {"type": "int", "constraints": ["unique", "non-null"]},
                              "post author": {"type": "str", "constraints": ["non-null", "alphanumeric"]},
                               "post title": {"type": "str"},
                              "comment author": {"type": "str", "constraints": ["non-null", "alphanumeric"]},
                              "comment body": {"type": "str", "allowed_values": ["[removed]"], "allow_empty": False},
                              "comment status": {"type": "str"},
                             "posted by": {"type": "str", "constraints": ["non-null", "alphanumeric"]},
                              "comment status": {"type": "str"},

                              "upvotes": {"type": "int", "constraints": ["non-negative"]},
                              "downvotes": {"type": "int", "constraints": ["non-negative"]},
                              "post score": {"type": "int", "constraints": ["non-negative"]},
                              "comment score": {"type": "int", "constraints": ["non-negative"]},
                              "score": {"type": "int", "constraints": ["non-negative"]},
                              "post date": {"type": "datetime", "constraints": ["valid-format"]},
                              "comment Date": {"type": "datetime", "constraints": ["valid-format"]},
                              "Timestamp": {"type": "datetime", "constraints": ["valid-format"]},
                              "Subreddit": {"type": "str", "constraints": ["non-null"]},
                          }
                      }
                  elif option == "Upload JSON Metadata File":
                       uploaded_file = st.file_uploader("Upload a JSON Metadata File", type="json")
    
                       if uploaded_file is not None:
                           metadata = json.load(uploaded_file)
                           #st.json(metadata)  # Display the uploaded metadata
                  if st.button('Check Consistency'):
                     if option == "Custom Metadata":
                        issues = check_consistency(df, custom_metadata)
                     elif option == "Upload JSON Metadata File":
                          issues = check_consistency(df, metadata)

    # Display results
                     if isinstance(issues, list) and issues:
                        st.error("Consistency Issues Found:")
                        for issue in issues:
                            st.write(f"- {issue}")
                     else:
                          st.success(issues)  # This will indicate that everything is consistent

                  #st.write(f" Nearly Redundant Post: {total_near_duplicates}")
               
                # Display the cleaned dataset
                



               with col4:
                    st.markdown("<span style='color:blue; font-weight:bold;'> Contextual Relevancy:</span>", unsafe_allow_html=True)
                         # User can input a query
                    query = st.text_input("Enter a keyword to search for:") 
                    if st.button("Search"):
                       if query:  
                          relevant_results = df[df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
                # Find irrelevant posts
                          irrelevant_results = df[~df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
                          revv=len(irrelevant_results)
                          if not  irrelevant_results.empty:
                             st.markdown("<span style='color:black; font-weight:bold;size=10px;'> Number of  Post:</span>", unsafe_allow_html=True)
                             #st.dataframe( irrelevant_results)
                             st.markdown(f"<p style='color: green;'>{revv}</p>", unsafe_allow_html=True)
                          else:
                             st.write("No irrelevant results found.")
                       else:
                          st.write("Please enter a keyword.")

                    #st.markdown("<span style='color:blue; font-weight:bold;'> Post Decay Rate:</span>", unsafe_allow_html=True)
                    st.markdown("<span style='color:blue; font-weight:bold;'> Post Decay Rate:</span>", unsafe_allow_html=True)
                    if 'post date' not in df.columns:
                        st.error("The 'Post Date' column is missing from the dataset.")
                    else:
    # Convert 'Post Date' to datetime, coerce invalid entries to NaT
                        df['post date'] = pd.to_datetime(df['post date'], errors='coerce')
    
    # Check for missing or invalid dates
                        if df['post date'].isnull().all():
                          st.error("All entries in the 'Post Date' column are invalid or missing.")
                        elif df['post date'].isnull().any():
                             st.warning(f"Some entries in the 'Post Date' column are invalid or missing ({df['post date'].isnull().sum()} rows).")
                        else:
                             df['Time Since Post'] = (df['post date'] - df['post date'].iloc[0]).dt.total_seconds() / 3600

    # Calculate and display post decay rate
                             calculate_decay_rate(df)

                    st.markdown("<span style='color:blue;font-weight:bold;'>Context Based Sentiment </span>", unsafe_allow_html=True)
                    if 'comment body' in df.columns or 'comment' in df.columns:
    # Identify the correct column dynamically
                        comment_column = 'comment body' if 'comment body' in df.columns else 'comment'
                        # Select N for N-grams
                        n = 2
                        summary = summarize_sentiments_and_frequencies(df, comment_column, n)
                        
                             #st.subheader("Sentiment Counts")
                        sentiment_counts = {
                         "Positive": summary["Positive"],
                         "Negative": summary["Negative"],
                         "Neutral": summary["Neutral"],
                            }
                       
                        st.write(sentiment_counts)
                    else:
                         st.error("The uploaded CSV does not contain a 'Comment' column.")

                 
                   
                    
                                
                   
                                
 
           
           with st.expander("", expanded=True):
                 col1, col2, col3, col4 = st.columns(4)

           with col1:
               
                
             st.subheader("Data Analysis Report")
         
             if st.button("Data Analysis Report"):
         # Generate the profiling report
                profile = ProfileReport(df, title="Data Quality Profile Report", explorative=True)

         # Save the report as an HTML file for download
                profile.to_file("data_quality_report.html")

         # Read the HTML content
                #with open('data_quality_report.html', 'r', encoding='utf-8') as f:
                    #  html_content = f.read()

         # Display the HTML report using Streamlit components
                #components.html(html_content, height=1000, scrolling=True)

         # Provide a download button for the report
                with open('data_quality_report.html', 'rb') as file:
                      st.download_button(
                           label="Download Data Analysis Report",
                 data=file,
                 file_name="data_quality_report.html",
                 mime="text/html"
              )

          
            
                
                
       
       
                   
                  
    else:
            st.write("Please upload a file to start monitoring social data.")
