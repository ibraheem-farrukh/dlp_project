import pandas as pd

# load dataset
df = pd.read_csv("data/output_ur.csv")

# print("Shape:", df.shape)
# print("Columns:", df.columns.tolist())

# # print("Dtypes:\n", df.dtypes)
# # print("Missing values:\n", df.isnull().sum())

# print("Describe:\n", df.describe())
# print("Sample data:\n", df.head())

# Explore dataset structure
print("\n=== Explore Dataset Structure ===")
print("Number of poems:", len(df))

print("\nSample poems:")
print(df.head(5))

# Assume poem text is in 'content' column; adjust if different
poem_col = 'content'

# Analyze poem lengths (characters)
df['poem_length_chars'] = df[poem_col].str.len()
print("\nPoem lengths (characters):")
print(df['poem_length_chars'].describe())

# Identify data quality issues
print("\nData Quality Issues:")
print("Missing values in poem column:", df[poem_col].isnull().sum())
print("Duplicate poems:", df.duplicated(subset=[poem_col]).sum())
print("Empty poems (length 0):", (df['poem_length_chars'] == 0).sum())

# Statistical analysis
print("\n=== Statistical Analysis ===")

# Average words per poem
df['word_count'] = df[poem_col].str.split().str.len()
print("Average words per poem:", df['word_count'].mean())

# Vocabulary richness
all_words = ' '.join(df[poem_col].dropna()).split()
unique_words = set(all_words)
total_words = len(all_words)
print("Vocabulary richness (unique/total words):", len(unique_words) / total_words if total_words > 0 else 0)

# Common words frequency
from collections import Counter
word_freq = Counter(all_words)
print("Top 10 common words:", word_freq.most_common(10))

# Length distribution (words)
print("\nLength distribution (words):")
print(df['word_count'].describe())

# Optional: plot histogram (requires matplotlib)
# import matplotlib.pyplot as plt
# plt.hist(df['word_count'], bins=20)
# plt.title('Poem Length Distribution (words)')
# plt.xlabel('Word Count')
# plt.ylabel('Frequency')
# plt.show()