import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

def plot_disease_graph(predictions):
    """Plots a bar graph for disease predictions."""
    # Count occurrences of each disease
    disease_counts = Counter(predictions.values())

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(disease_counts.keys(), disease_counts.values(), color='skyblue')
    plt.title("Disease Prediction Counts", fontsize=16)
    plt.xlabel("Disease", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the graph in Streamlit
    st.pyplot(plt)
