import tkinter as tk
from tkinter import messagebox, scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    """Calculate similarity using character n-grams (standard method for style detection)"""
    if not text1 or not text2:
        return None
    
    if len(text1.split()) < 10 or len(text2.split()) < 10:
        messagebox.showwarning("Warning", "Texts should have at least 10 words for accurate analysis")
    
    # Character n-gram TfidfVectorizer
    # analyzer='char' looks at character patterns, not words
    # ngram_range=(2,4) captures 2, 3, and 4 character sequences
    # This captures writing style independent of vocabulary
    vectorizer = TfidfVectorizer(
        analyzer='char',      # Use characters instead of words
        ngram_range=(2, 4),   # Look at 2, 3, and 4 character patterns
        lowercase=True
    )
    
    try:
        # Create TF-IDF vectors based on character patterns
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    except:
        return 0.0

def interpret_similarity(score):
    """Interpret the similarity score"""
    if score >= 0.70:
        return "Very Similar - Likely Same Author"
    elif score >= 0.50:
        return "Somewhat Similar - Possibly Same Author or Related Topics"
    else:
        return "Different - Likely Different Authors or Topics"

def compare_texts():
    """Main function to compare the two text inputs"""
    text1 = text_box1.get("1.0", tk.END).strip()
    text2 = text_box2.get("1.0", tk.END).strip()
    
    if not text1 or not text2:
        messagebox.showerror("Error", "Please enter text in both boxes")
        return
    
    similarity_score = calculate_similarity(text1, text2)
    
    if similarity_score is not None:
        interpretation = interpret_similarity(similarity_score)
        
        result_text = f"Similarity Score: {similarity_score:.4f}\n\n"
        result_text += f"Interpretation: {interpretation}\n\n"
        result_text += f"Word Count - Text 1: {len(text1.split())} words\n"
        result_text += f"Word Count - Text 2: {len(text2.split())} words"
        
        result_label.config(text=result_text)

def clear_all():
    """Clear all text boxes and results"""
    text_box1.delete("1.0", tk.END)
    text_box2.delete("1.0", tk.END)
    result_label.config(text="Results will appear here")

# Create main window
root = tk.Tk()
root.title("AI Writing Style Detector")
root.geometry("900x600")
root.configure(bg="#f0f0f0")

# Title
title_label = tk.Label(root, text="AI Writing Style Detector", 
                       font=("Arial", 20, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

# Frame for text boxes
text_frame = tk.Frame(root, bg="#f0f0f0")
text_frame.pack(pady=10, padx=20)

# Text Box 1
label1 = tk.Label(text_frame, text="Text Sample 1", font=("Arial", 12, "bold"), bg="#f0f0f0")
label1.grid(row=0, column=0, padx=10)

text_box1 = scrolledtext.ScrolledText(text_frame, width=40, height=15, 
                                       font=("Arial", 10), wrap=tk.WORD)
text_box1.grid(row=1, column=0, padx=10)

# Text Box 2
label2 = tk.Label(text_frame, text="Text Sample 2", font=("Arial", 12, "bold"), bg="#f0f0f0")
label2.grid(row=0, column=1, padx=10)

text_box2 = scrolledtext.ScrolledText(text_frame, width=40, height=15, 
                                       font=("Arial", 10), wrap=tk.WORD)
text_box2.grid(row=1, column=1, padx=10)

# Buttons frame
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

compare_button = tk.Button(button_frame, text="Compare Texts", 
                           command=compare_texts, font=("Arial", 12, "bold"),
                           bg="#4CAF50", fg="white", width=15)
compare_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear All", 
                         command=clear_all, font=("Arial", 12, "bold"),
                         bg="#f44336", fg="white", width=15)
clear_button.grid(row=0, column=1, padx=10)

# Results frame
result_frame = tk.Frame(root, bg="white", relief=tk.RAISED, borderwidth=2)
result_frame.pack(pady=10, padx=20, fill=tk.BOTH)

result_label = tk.Label(result_frame, text="Results will appear here", 
                        font=("Arial", 11), bg="white", justify=tk.LEFT,
                        padx=20, pady=20)
result_label.pack()

# Instructions
instructions = "Instructions: Paste two text samples (at least 10 words each) and click 'Compare Texts'"
instruction_label = tk.Label(root, text=instructions, font=("Arial", 9), 
                            bg="#f0f0f0", fg="#666")
instruction_label.pack(pady=5)

# Run the application
root.mainloop()