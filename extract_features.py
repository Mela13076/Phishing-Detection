import pandas as pd
from urllib.parse import urlparse, parse_qs
import re

# extract features function
def extract_features(url):
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    
    # Check for IP address in the domain
    ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    is_ip_address = 1 if ip_pattern.match(parsed_url.netloc) else 0
    
    # Common phishing keywords
    phishing_keywords = ['login', 'verify', 'account', 'update', 'banking']
    keywords_count = sum(word in parsed_url.path.lower() for word in phishing_keywords)

    # Check for URL shortening services
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
    is_shortened = any(shortener in parsed_url.netloc for shortener in shorteners)

    # return all extracted features from URL 
    return {
        'domain': parsed_url.netloc,
        'path_length': len(parsed_url.path),
        'use_https': 1 if parsed_url.scheme == 'https' else 0,
        'num_subdomains': len(domain_parts) - 2 if len(domain_parts) > 2 else 0,
        'is_ip_address': is_ip_address,
        'token_count_path': len(parsed_url.path.split('/')),
        'contains_suspicious_keywords': keywords_count,
        'url_length':  len(url),
        'query_count': len(parse_qs(parsed_url.query)),
        'is_shortened': int(is_shortened),
        'has_at_symbol': int('@' in url),
        'count_hyphens': url.count('-')
    }

# Load URLs from a text file and label them
def load_urls(filename, label):
    with open(filename, 'r') as file:
        urls = file.read().splitlines()
    return pd.DataFrame({
        'url': urls,
        'label': [label] * len(urls)
    })

# Loading data form text files
phishing_data = load_urls('phishing_urls_train.txt', 'phishing')
legitimate_data = load_urls('legitimate_urls_train.txt', 'legitimate')

# Combine phishing and legitimate data
combined_data = pd.concat([phishing_data, legitimate_data], ignore_index=True)

# Apply the extract_features function to each URL and expand the returned dicts into separate columns
features = combined_data['url'].apply(extract_features)
features_df = pd.json_normalize(features)
combined_data = pd.concat([combined_data, features_df], axis=1)

# Save the DataFrame with URLs, labels, and features to a CSV file
combined_data.to_csv('urls_train2.csv', index=False)

print("Data with features saved to CSV file.")
